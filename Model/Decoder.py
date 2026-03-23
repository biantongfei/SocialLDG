from torch import nn
import torch

from Data.constants import coco_body_point_num, head_point_num, hands_point_num


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len, time_layers, num_heads):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = coco_body_point_num + head_point_num + hands_point_num

        self.seq_len = seq_len
        self.output_dim = (coco_body_point_num + head_point_num + hands_point_num) * 2  # 只重建 x, y

        self.query_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))  # Learnable Queries (类似 DETR)

        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 2,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=time_layers)
        self.head = nn.Linear(latent_dim, self.output_dim)

    def forward(self, z):
        B = z.shape[0]

        # Expand query
        tgt = self.query_embed.expand(B, -1, -1)

        memory = z.unsqueeze(1)

        out_seq = self.transformer_decoder(tgt, memory)
        pred = self.head(out_seq)  # [B, 10, 133*2]

        return pred.view(B, self.seq_len, self.num_joints, 2)
