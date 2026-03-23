from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from Data.constants import coco_body_point_num, device, head_point_num, hands_point_num, coco_body_l_pair, head_l_pair, \
    hand_l_pair


class Pose_Encoder(nn.Module):
    def __init__(self,
                 batch_size,
                 sequence_length,
                 encoder_gcn_layers,
                 encoder_time_layers,
                 keypoint_hidden_dim,
                 num_heads,
                 representation_length,
                 dropout=0,
                 ):
        super(Pose_Encoder, self).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.representation_length = representation_length
        self.input_size = coco_body_point_num + head_point_num + hands_point_num
        self.dropout_rate = dropout

        curr_dim = 3
        self.body_graph_layers = nn.ModuleList()
        self.body_graph_norms = nn.ModuleList()
        self.head_graph_layers = nn.ModuleList()
        self.head_graph_norms = nn.ModuleList()
        self.hand_graph_layers = nn.ModuleList()
        self.hand_graph_norms = nn.ModuleList()
        for i in range(encoder_gcn_layers):
            body_layer = GATConv(curr_dim, keypoint_hidden_dim, heads=num_heads, concat=True,
                                 add_self_loops=False,
                                 dropout=dropout)
            head_layer = GATConv(curr_dim, keypoint_hidden_dim, heads=num_heads, concat=True,
                                 add_self_loops=False,
                                 dropout=dropout)
            hand_layer = GATConv(curr_dim, keypoint_hidden_dim, heads=num_heads, concat=True,
                                 add_self_loops=False,
                                 dropout=dropout)
            curr_dim = keypoint_hidden_dim * num_heads
            self.body_graph_layers.append(body_layer)
            self.body_graph_norms.append(nn.LayerNorm(curr_dim))
            self.head_graph_layers.append(head_layer)
            self.head_graph_norms.append(nn.LayerNorm(curr_dim))
            self.hand_graph_layers.append(hand_layer)
            self.hand_graph_norms.append(nn.LayerNorm(curr_dim))

        body_edge_index = torch.Tensor(coco_body_l_pair).t().to(device)
        body_edge_index = to_undirected(body_edge_index)
        offsets = torch.arange(batch_size * sequence_length, device=device) * coco_body_point_num
        edges_expanded = body_edge_index.unsqueeze(1)
        offsets_expanded = offsets.view(1, batch_size * sequence_length, 1)
        body_edge_index = edges_expanded + offsets_expanded
        self.body_edge_index = body_edge_index.reshape(2, -1).to(torch.long)

        head_edge_index = torch.Tensor(head_l_pair).t().to(device) - coco_body_point_num
        head_edge_index = to_undirected(head_edge_index)
        offsets = torch.arange(batch_size * sequence_length, device=device) * head_point_num
        edges_expanded = head_edge_index.unsqueeze(1)
        offsets_expanded = offsets.view(1, batch_size * sequence_length, 1)
        head_edge_index = edges_expanded + offsets_expanded
        self.head_edge_index = head_edge_index.reshape(2, -1).to(torch.long)

        hand_edge_index = torch.Tensor(hand_l_pair).t().to(device) - coco_body_point_num - head_point_num
        hand_edge_index = to_undirected(hand_edge_index)
        offsets = torch.arange(batch_size * sequence_length, device=device) * hands_point_num
        edges_expanded = hand_edge_index.unsqueeze(1)
        offsets_expanded = offsets.view(1, batch_size * sequence_length, 1)
        hand_edge_index = edges_expanded + offsets_expanded
        self.hand_edge_index = hand_edge_index.reshape(2, -1).to(torch.long)

        self.spatial_proj = nn.Sequential(
            nn.Linear(curr_dim * 3, representation_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, representation_length))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_length + 1, representation_length))
        encoder_layer = nn.TransformerEncoderLayer(d_model=representation_length,
                                                   nhead=representation_length,
                                                   dim_feedforward=representation_length * 2,
                                                   dropout=dropout, batch_first=True)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_time_layers)

    def forward(self, data):
        B, T, _, C = data[0].shape
        x_body, x_head, x_hand = data[0], data[1], data[2]
        curr_x_body = x_body.view(-1, C)  # [B*T*V, C]
        curr_x_head = x_head.view(-1, C)  # [B*T*V, C]
        curr_x_hand = x_hand.view(-1, C)  # [B*T*V, C]

        if B == self.batch_size:
            batch_body_edge_index = self.body_edge_index
            batch_head_edge_index = self.head_edge_index
            batch_hand_edge_index = self.hand_edge_index
        else:
            batch_body_edge_index = self.body_edge_index[:, :B * len(coco_body_l_pair)]
            batch_head_edge_index = self.head_edge_index[:, :B * len(head_l_pair)]
            batch_hand_edge_index = self.hand_edge_index[:, :B * len(hand_l_pair)]
        body_adj = SparseTensor(
            row=batch_body_edge_index[0],
            col=batch_body_edge_index[1],
            sparse_sizes=(curr_x_body.size(0), curr_x_body.size(0))
        ).set_diag()
        head_adj = SparseTensor(
            row=batch_head_edge_index[0],
            col=batch_head_edge_index[1],
            sparse_sizes=(curr_x_head.size(0), curr_x_head.size(0))
        ).set_diag()
        hand_adj = SparseTensor(
            row=batch_hand_edge_index[0],
            col=batch_hand_edge_index[1],
            sparse_sizes=(curr_x_hand.size(0), curr_x_hand.size(0))
        ).set_diag()

        for i, (body_graph_layer, head_graph_layer, hand_graph_layer) in enumerate(
                zip(self.body_graph_layers, self.head_graph_layers, self.hand_graph_layers)):
            curr_x_body = body_graph_layer(curr_x_body, body_adj)
            curr_x_body = self.body_graph_norms[i](curr_x_body)
            curr_x_body = F.elu(curr_x_body)
            curr_x_body = F.dropout(curr_x_body, p=self.dropout_rate, training=self.training)
            curr_x_head = head_graph_layer(curr_x_head, head_adj)
            curr_x_head = self.head_graph_norms[i](curr_x_head)
            curr_x_head = F.elu(curr_x_head)
            curr_x_head = F.dropout(curr_x_head, p=self.dropout_rate, training=self.training)
            curr_x_hand = hand_graph_layer(curr_x_hand, hand_adj)
            curr_x_hand = self.hand_graph_norms[i](curr_x_hand)
            curr_x_hand = F.elu(curr_x_hand)
            curr_x_hand = F.dropout(curr_x_hand, p=self.dropout_rate, training=self.training)

        # Node -> Frame Pooling
        x_body = curr_x_body.view(B * T, coco_body_point_num, -1)
        x_head = curr_x_head.view(B * T, head_point_num, -1)
        x_hand = curr_x_hand.view(B * T, hands_point_num, -1)
        x = torch.cat((x_body.mean(dim=1), x_head.mean(dim=1), x_hand.mean(dim=1)), dim=1)
        x = self.spatial_proj(x)  # [B*T, 3*H]
        x = x.view(B, T, -1)  # [B, T, H]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        token_seq = torch.cat((cls_tokens, x), dim=1)
        token_seq = token_seq + self.pos_embedding
        out_seq = self.trans_encoder(token_seq)
        x = out_seq[:, 0, :]  # [B, H]
        return x
