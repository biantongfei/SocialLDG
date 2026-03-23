import torch
from torch.nn import functional
import argparse
from collections import OrderedDict
from Model import Pose_AutoEncoder
from Data.Dataset import get_datasets
from Data.DataLoader import SocialSDG_DataLoader
from scripts import load_config
from Data.constants import device, coco_body_point_num, head_point_num


def parse_args():
    parser = argparse.ArgumentParser(description='Test Whole-Body Pose Sequence AutoEncoder')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    return parser.parse_args()


def test_autoencoder(args, config):
    net = Pose_AutoEncoder(batch_size=config['train']['batch_size'],
                           sequence_length=config['data']['sequence_length'],
                           encoder_gcn_layers=config['model']['encoder_gcn_layers'],
                           encoder_time_layers=config['model']['encoder_time_layers'],
                           keypoint_hidden_dim=config['model']['keypoint_hidden_dim'],
                           num_heads=config['model']['num_heads'],
                           representation_length=config['model']['representation_length'],
                           decoder_time_layers=config['model']['decoder_time_layers'],
                           dropout=config['train']['dropout'])
    weights = torch.load(args.checkpoint_path)
    weights = OrderedDict([[k, v.cuda(device)] for k, v in weights.items()])
    net.load_state_dict(weights, strict=True)
    net.to(device)
    head_loss_boost = 1.5  # 提升面部的权重
    hand_loss_boost = 2.0  # 提升手部的权重，平衡低置信度

    net.eval()
    testset = get_datasets(
        data_path=args.data_path,
        sequence_length=config['data']['sequence_length'],
        future_length=config['data']['future_length'],
        test=True)
    test_loader = SocialSDG_DataLoader(
        dataset=testset,
        batch_size=config['train']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        shuffle=False,
        drop_last=False,
        zero_mask_rate=config['train']['zero_mask_rate'])
    total_loss, total_mae = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, _, targets = data
            confidence_score = targets[:, :, :, 2]
            confidence_score = confidence_score.clamp(min=0.01)
            confidence_score[:, :, coco_body_point_num:coco_body_point_num + head_point_num] *= head_loss_boost
            confidence_score[:, :, coco_body_point_num + head_point_num:] *= hand_loss_boost
            targets = targets[:, :, :, :2]
            outputs = net(inputs)
            mae = functional.l1_loss(outputs, targets, reduction='none')
            loss = functional.mse_loss(outputs, targets, reduction='none')
            loss *= confidence_score.unsqueeze(-1)
            loss = loss.sum() / confidence_score.sum()
            total_loss += loss.item() * targets.shape[0]
            total_mae += mae.mean().item() * targets.shape[0]
    print('testing--> test_loss: %.6f, test_mae: %.6f' % (total_loss / len(testset), total_mae / len(testset)))
    return


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    test_autoencoder(args, config)
