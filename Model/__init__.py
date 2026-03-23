from torch import nn

from Model.Encoder import Pose_Encoder
from Model.Decoder import TransformerDecoder
from Model.SocialLDG import SocialLDG
from Data.constants import original_subtasks


class Pose_AutoEncoder(nn.Module):
    def __init__(self,
                 batch_size,
                 sequence_length,
                 encoder_gcn_layers,
                 encoder_time_layers,
                 keypoint_hidden_dim,
                 num_heads,
                 representation_length,
                 decoder_time_layers=2,
                 dropout=0,
                 ):
        super(Pose_AutoEncoder, self).__init__()
        self.encoder = Pose_Encoder(
            batch_size=batch_size,
            sequence_length=sequence_length,
            encoder_gcn_layers=encoder_gcn_layers,
            encoder_time_layers=encoder_time_layers,
            keypoint_hidden_dim=keypoint_hidden_dim,
            num_heads=num_heads,
            representation_length=representation_length,
            dropout=dropout)

        self.decoder = TransformerDecoder(
            latent_dim=representation_length,
            seq_len=sequence_length,
            time_layers=decoder_time_layers,
            num_heads=num_heads)

    def forward(self, data):
        x = self.encoder(data)
        out = self.decoder(x)
        return out


class Encoder_SocialLDG(nn.Module):
    def __init__(self,
                 batch_size,
                 sequence_length,
                 encoder_gcn_layers,
                 encoder_time_layers,
                 keypoint_hidden_dim,
                 num_heads,
                 representation_length,
                 dropout=0,
                 hidden_dim=128,
                 n_heads=2,
                 msg_pass_steps=1,
                 task_token='scibert',
                 subtasks=original_subtasks
                 ):
        super(Encoder_SocialLDG, self).__init__()
        self.encoder = Pose_Encoder(
            batch_size=batch_size,
            sequence_length=sequence_length,
            encoder_gcn_layers=encoder_gcn_layers,
            encoder_time_layers=encoder_time_layers,
            keypoint_hidden_dim=keypoint_hidden_dim,
            num_heads=num_heads,
            representation_length=representation_length,
            dropout=dropout)

        self.classifier = SocialLDG(
            z_dim=representation_length,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            msg_pass_steps=msg_pass_steps,
            task_token=task_token,
            subtasks=subtasks)

    def forward(self, data):
        x = self.encoder(data)
        out = self.classifier(x)
        out = out['preds'], (out['edge_index'], out['edge_weights']), out['edge_regularization']
        return out
