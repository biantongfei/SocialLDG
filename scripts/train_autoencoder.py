import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from tqdm import tqdm
import torch
from torch.nn import functional

from Data.Dataset import get_datasets
from Data.DataLoader import get_dataloaders
from Model import Pose_AutoEncoder
from scripts import EarlyStopping, load_config
from Data.constants import device, coco_body_point_num, head_point_num


def parse_args():
    parser = argparse.ArgumentParser(description='Train Whole-Body Pose Sequence AutoEncoder')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_weights', type=str, default=None)
    return parser.parse_args()


def train_autoencoder(args, config):
    MAX_EPOCH = 100
    head_loss_boost = 1.5  # 提升面部的权重
    hand_loss_boost = 2.0  # 提升手部的权重，平衡低置信度
    net = Pose_AutoEncoder(batch_size=config['train']['batch_size'],
                           sequence_length=config['data']['sequence_length'],
                           encoder_gcn_layers=config['model']['encoder_gcn_layers'],
                           encoder_time_layers=config['model']['encoder_time_layers'],
                           keypoint_hidden_dim=config['model']['keypoint_hidden_dim'],
                           num_heads=config['model']['num_heads'],
                           representation_length=config['model']['representation_length'],
                           decoder_time_layers=config['model']['decoder_time_layers'],
                           dropout=config['train']['dropout'])
    net.to(device)
    if config['train']['epoch'] == 'early_stop':
        print("Early Stop is ON")
        early_stopper = EarlyStopping(delta=1e-3)
    else:
        print("Early Stop is OFF")
        early_stopper = None
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in list(net.named_parameters()) if not any(nd in n for nd in no_decay)],
         'weight_decay': config['train']['weight_decay']},
        {'params': [p for n, p in list(net.named_parameters()) if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}], lr=config['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-7,
        threshold=1e-4,
        threshold_mode='abs'
    )
    WARMUP_EPOCHS = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    trainset, valset, testset = get_datasets(
        data_path=args.data_path,
        sequence_length=config['data']['sequence_length'],
        future_length=config['data']['future_length'])
    train_loader, val_loader, test_loader = get_dataloaders(
        trainset,
        valset,
        testset,
        sequence_length=config['data']['sequence_length'],
        batch_size=config['train']['batch_size'],
        zero_mask_rate=config['train']['zero_mask_rate'])
    for epoch in range(MAX_EPOCH):
        net.train()
        print('Training')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        total_loss, total_mae = 0, 0
        for i, data in enumerate(train_loader):
            progress_bar.update(1)
            optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.shape[0]
            total_mae += mae.mean().item() * targets.shape[0]
        progress_bar.close()
        result_str = 'training-->  epoch: %d, ' % epoch
        print(
            result_str + 'train_loss: %.6f, train_mae: %.6f' % (total_loss / len(trainset), total_mae / len(trainset)))
        print('Validating')
        total_loss, total_mae = 0, 0
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
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
        result_str = 'validating--> epoch: %d, ' % epoch
        print(result_str + 'val_loss: %.6f, val_mae: %.6f' % (total_loss / len(valset), total_mae / len(valset)))
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Warmup Phase: LR = {current_lr:.6f}")
        else:
            scheduler.step(total_mae / len(testset))
        if config['train']['epoch'] == 'early_stop':
            if args.save_weights and early_stopper.best_score > total_loss / len(valset):
                model_name = 'new_weights/autoencoder_rl%d_zmr%d.pt' % (config['data'].representation_length,
                                                                        int(config['train'].zero_mask_rate * 100))
                torch.save(net.state_dict(), model_name)
            early_stopper(total_mae / len(testset))
            print('The Best Score is %.6f, %d more epoch to go.' % (
                early_stopper.best_score, early_stopper.patience - early_stopper.counter))
            if early_stopper.early_stop:
                print(
                    "Early stopping triggered, stopping training. The best score is %.6f" % early_stopper.best_score)
                break
        elif epoch == config['train']['epochs'] - 1:
            if args.save_weights:
                model_name = 'new_weights/autoencoder_rl%d_zmr%d.pt' % (config['data'].representation_length,
                                                                        int(config['train'].zero_mask_rate * 100))
                torch.save(net.state_dict(), model_name)
            break
        else:
            print('------------------------------------------')
            # break
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Testing')
    total_loss, total_mae = 0, 0
    net.eval()
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
    result_str = 'testing--> epoch: %d, ' % epoch
    print(result_str + 'test_loss: %.6f, test_mae: %.6f' % (total_loss / len(testset), total_mae / len(testset)))
    return


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    train_autoencoder(args, config)
