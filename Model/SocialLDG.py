from torch import nn
import torch

import math

from Data.constants import device, contact_classes, intention_classes, attitude_classes, jpl_harper_action_classes, \
    original_subtasks


class TaskPromptInjection(nn.Module):
    """
    Add token semantics as a ‘Prompt’ to the Backbone feature, endowing nodes with task specificity.
    """

    def __init__(self, z_dim, token_dim=512, hidden_dim=64):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, token_embeddings, backbone_feature):
        num_tasks = token_embeddings.size(0)
        task_tokens = self.projector(token_embeddings).unsqueeze(0)
        z_expanded = self.z_proj(backbone_feature).unsqueeze(1)
        return z_expanded.expand(-1, num_tasks, -1) + task_tokens


class SemanticBilinearEdgeGenerator(nn.Module):
    def __init__(self, task_token_dim, z_dim, n_heads):
        super().__init__()
        inter_dim = int(z_dim / 2)
        self.n_heads = n_heads
        # Map the task token to the Query/Key space used for calculating edge bias.
        self.w_query = nn.Linear(task_token_dim, inter_dim * n_heads)
        self.w_key = nn.Linear(task_token_dim, inter_dim * n_heads)

        # Context Modulator: Maps Backbone features to semantic activation vectors
        self.context_modulator = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, inter_dim * n_heads),
        )

        # Final scaling factor
        self.scale = inter_dim ** -0.5
        # Each head has its own independent temperature setting; some heads can be very sharp, while others can be very smooth.
        self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, backbone_feat, clip_embeddings):
        B = backbone_feat.size(0)
        T = clip_embeddings.size(0)

        Q = self.w_query(clip_embeddings).view(T, self.n_heads, -1)
        K = self.w_key(clip_embeddings).view(T, self.n_heads, -1)

        context = self.context_modulator(backbone_feat).view(B, self.n_heads, -1)

        # Query (or Key) Modulation 1.0 ensures preservation of original semantics while performing context perturbation.
        Q_modulated = Q.unsqueeze(0) * (1.0 + torch.tanh(context.unsqueeze(1)))

        Q_permuted = Q_modulated.permute(0, 2, 1, 3)  # [B, H, T, Inter]
        K_permuted = K.permute(1, 0, 2)  # [H, T, Inter]

        # Calculate Bilinear Matching Scores (Batch Matrix Multiplication)
        edge_logits = torch.matmul(Q_permuted, K_permuted.transpose(-1, -2)) * self.scale

        return edge_logits / torch.clamp(self.temperature, min=0.01)


class ResidualTaskHead(nn.Module):
    """
    Receive SocialLDG Output and Raw Encoder Output, then fuse and classify.
    """

    def __init__(self, z_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, graph_node_feat, raw_z):
        combined = torch.cat([graph_node_feat, raw_z], dim=-1)
        return self.fc(combined)


# ---------------------------
# SocialLDG
# ---------------------------
class SocialLDG(nn.Module):
    def __init__(
            self,
            z_dim: int,
            hidden_dim: int,
            n_heads: int = 4,
            msg_pass_steps: int = 1,
            task_token: str = 'scibert',
            dropout: float = 0.1,
            subtasks: list = original_subtasks
    ):
        """
        z_dim: backbone 输出维度（每样本）
        task_token_dim: 每任务的 token 维度
        node_dim: 每个节点表示维度 H
        n_heads: multi-head attention heads
        msg_pass_steps: 消息传递迭代次数 K
        """
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.T = len(subtasks)
        self.task_token_dim = 768
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "node_dim must be divisible by n_heads"
        self.K = msg_pass_steps
        self.subtasks = subtasks

        # Disconnect the link from future tasks to current tasks
        mask_tensor = torch.ones((len(subtasks), len(subtasks)))
        for i, t_i in enumerate(subtasks):
            if 'current' in t_i:
                for j, t_j in enumerate(subtasks):
                    if 'future' in t_j:
                        mask_tensor[i, j] = 0

        self.register_buffer("edge_mask", mask_tensor.to(device))
        self.edge_weights = None
        self.edge_bias = None

        # Learnable Task Token
        if task_token == 'scibert':
            task_tokens = torch.load('TaskToken/scibert_centered_task_tokens.pt')
            self.task_token_dim = 768
        elif task_token == 'clip':
            task_tokens = torch.load('TaskToken/clip_centered_task_tokens.pt')
            self.task_token_dim = 512
        elif task_token == 'bert':
            task_tokens = torch.load('TaskToken/bert_centered_task_tokens.pt')
            self.task_token_dim = 768
        elif task_token.startswith('sbert'):
            task_tokens = torch.load('TaskToken/sbert_centered_task_tokens.pt')
            self.task_token_dim = 384
        elif task_token.startswith('st5'):
            task_tokens = torch.load('TaskToken/st5_centered_task_tokens.pt')
            self.task_token_dim = 768


        elif task_token == 'random':
            task_tokens = nn.Parameter(torch.randn(self.T, self.task_token_dim))

        task_tokens = self.slicing_task_tokens(task_tokens)
        print(f"Loaded {task_token} tokens, shape: {task_tokens.shape}")
        self.task_tokens = nn.Parameter(task_tokens, requires_grad=True)

        # The adapter for each task (mapping z + token to node feature)
        self.task_prompt_injection = TaskPromptInjection(z_dim=z_dim, hidden_dim=hidden_dim,
                                                         token_dim=self.task_token_dim)
        # edge gate generation: z -> (T*T) logits
        self.edge_gen = SemanticBilinearEdgeGenerator(z_dim=z_dim, n_heads=n_heads, task_token_dim=self.task_token_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.node_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # 每任务的 head
        self.heads = nn.ModuleList()
        if 'contact_current' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(contact_classes), dropout=dropout))
        if 'contact_future' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(contact_classes), dropout=dropout))
        if 'intention' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(intention_classes),
                                 dropout=dropout))
        if 'attitude' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(attitude_classes), dropout=dropout))
        if 'action_current' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(jpl_harper_action_classes),
                                 dropout=dropout))
        if 'action_future' in subtasks:
            self.heads.append(
                ResidualTaskHead(z_dim=z_dim, hidden_dim=hidden_dim, output_dim=len(jpl_harper_action_classes),
                                 dropout=dropout))

    def forward(self, z: torch.Tensor):
        B = z.shape[0]
        T = self.T

        # 1) Fuse the backbone's output z with the task token.
        h = self.task_prompt_injection(self.task_tokens, z)  # (B, node_dim)

        # 2) edge bias: z -> (B, T, T)
        edge_bias = self.edge_gen(z, self.task_tokens)  # (B, T*T)

        # 3) Single-step messaging
        alpha_steps = []
        for step in range(self.K):
            h_norm = self.norm1(h)
            q = self.W_q(h_norm)  # (B,T,H)
            k = self.W_k(h_norm)
            v = self.W_v(h_norm)

            # reshape for multi-head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
            qh = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            kh = k.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            vh = v.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # scores = q @ k^T / sqrt
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,heads,T,T)

            # Combine heads' scores
            scores = scores + edge_bias

            # sigmoid over source nodes j for each target i: that is, for fixed i, sum_j alpha_{i<-j} = 1
            alpha = torch.nn.functional.sigmoid(scores)  # (B,heads,T,T)

            # apply gate multiplicatively to scores (directed: i <- j uses row i, col j)
            mask_broadcast = self.edge_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            alpha = alpha.masked_fill(mask_broadcast == 0, 0.0)

            alpha_steps.append(alpha.detach().mean(dim=1))  # store mean over heads for interpretation (B,T,T)

            # attention output: (B,heads,T,head_dim) <- alpha @ v
            attn_out = torch.matmul(alpha, vh)  # (B,heads,T,head_dim)

            # merge heads back: (B,T,H)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, self.hidden_dim)

            # Residual 1
            h = h + self.dropout(attn_out)
            h = h + self.node_ffn(self.norm2(h))

        # 4) heads -> preds
        preds = []
        for i, head in enumerate(self.heads):
            preds.append(head(h[:, i, :], z))
        l1_loss = alpha_steps[0].mean()
        self.edge_bias = edge_bias
        self.edge_weights = alpha_steps[0]
        return {"preds": tuple(preds), "edge_index": self.edge_mask.nonzero(as_tuple=False),
                "edge_weights": alpha_steps[0], "edge_regularization": l1_loss, "z": z}

    def slicing_task_tokens(self, task_tokens):
        sliced_task_tokens = torch.zeros((len(self.subtasks), task_tokens.shape[1]))
        for i, task in enumerate(self.subtasks):
            sliced_task_tokens[i] = task_tokens[original_subtasks.index(task)]
        return sliced_task_tokens

    def expand_task_tokens(self, task_token):
        if task_token.startswith('clip'):
            full_task_tokens = torch.load('TaskToken/clip_centered_task_tokens.pt')
            print(f"Loaded CLIP tokens, shape: {full_task_tokens.shape}")
        elif task_token.startswith('sbert'):
            full_task_tokens = torch.load('TaskToken/sbert_centered_task_tokens.pt')
            print(f"Loaded SBERT tokens, shape: {full_task_tokens.shape}")
        elif task_token.startswith('scibert'):
            full_task_tokens = torch.load('TaskToken/scibert_centered_task_tokens.pt')
            print(f"Loaded SciBERT tokens, shape: {full_task_tokens.shape}")
        elif task_token.startswith('st5'):
            full_task_tokens = torch.load('TaskToken/st5_centered_task_tokens.pt')
            print(f"Loaded ST5 tokens, shape: {full_task_tokens.shape}")
        full_task_tokens = nn.Parameter(full_task_tokens, requires_grad=True)
        tokens_list = []
        for i, task in enumerate(original_subtasks):
            if task in self.subtasks:
                tokens_list.append(self.task_tokens[self.subtasks.index(task)].unsqueeze(0))
            else:
                tokens_list.append(full_task_tokens[i].unsqueeze(0))
        self.task_tokens = nn.Parameter(torch.cat(tokens_list, dim=0), requires_grad=True)
