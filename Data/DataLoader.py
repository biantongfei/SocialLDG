from torch.utils.data import DataLoader
import torch

import random

from Data.constants import coco_body_point_num, head_point_num, hands_point_num, device, dtype


class SocialLDG_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sequence_length, shuffle, drop_last, zero_mask_rate=0):
        super(SocialLDG_DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                   drop_last=drop_last)
        self.collate_fn = self.socialSDG_collate_fn
        self.sequence_length = sequence_length
        self.zero_mask_rate = zero_mask_rate

    def socialSDG_collate_fn(self, data):
        x_tensors_list = [
            torch.zeros((len(data), self.sequence_length, coco_body_point_num, 3), dtype=dtype, device=device),
            torch.zeros((len(data), self.sequence_length, head_point_num, 3), dtype=dtype, device=device),
            torch.zeros((len(data), self.sequence_length, hands_point_num, 3), dtype=dtype, device=device)]
        contact_cur_labels, contact_fut_labels, intention_labels, attitude_labels, action_cur_labels, action_fut_labels, interaction_stage_labels, video_labels, user_id_labels = [], [], [], [], [], [], [], [], []
        y_tensors = torch.zeros(
            (len(data), self.sequence_length, coco_body_point_num + head_point_num + hands_point_num, 3), dtype=dtype,
            device=device)
        for i, (features, labels) in enumerate(data):
            for ii in range(len(features)):
                if type(features[ii]) != int:
                    x = features[ii]
                    if self.zero_mask_rate > 0:
                        x = self.add_zero_mask(x)
                    x_tensors_list[ii][i] = x
            y_tensors[i] = torch.cat(features, dim=1)
            contact_cur_labels.append(labels[0][0])
            contact_fut_labels.append(labels[0][1])
            intention_labels.append(labels[0][2])
            attitude_labels.append(labels[0][3])
            action_cur_labels.append(labels[0][4])
            action_fut_labels.append(labels[0][5])
            interaction_stage_labels.append(labels[1][0])
            video_labels.append(labels[1][1])
        return x_tensors_list, (
            (torch.Tensor(contact_cur_labels), torch.Tensor(contact_fut_labels), torch.Tensor(intention_labels),
             torch.Tensor(attitude_labels), torch.Tensor(action_cur_labels), torch.Tensor(action_fut_labels)),
            (torch.Tensor(interaction_stage_labels), video_labels)), y_tensors

    def add_zero_mask(self, pose_sequence):
        if random.random() < 0.5:
            mask = torch.rand(pose_sequence.shape[0], pose_sequence.shape[1],
                              device=pose_sequence.device) > self.zero_mask_rate
            mask = mask.unsqueeze(-1)  # [T, N, 1]
            pose_sequence = pose_sequence * mask
        return pose_sequence


def get_dataloaders(trainset, valset, testset, sequence_length=10, batch_size=128, zero_mask_rate=0):
    train_dataloader = SocialLDG_DataLoader(dataset=trainset, batch_size=batch_size, sequence_length=sequence_length,
                                            shuffle=True, drop_last=False, zero_mask_rate=zero_mask_rate)
    val_dataloader = SocialLDG_DataLoader(dataset=valset, batch_size=batch_size, sequence_length=sequence_length,
                                          shuffle=False, drop_last=False, zero_mask_rate=zero_mask_rate)
    test_dataloader = SocialLDG_DataLoader(dataset=testset, batch_size=batch_size, sequence_length=sequence_length,
                                           shuffle=False, drop_last=False, zero_mask_rate=zero_mask_rate)
    return train_dataloader, val_dataloader, test_dataloader
