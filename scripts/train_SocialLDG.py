import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from torch.nn import functional

from tqdm import tqdm
from collections import OrderedDict
import math
import argparse

from Model import Encoder_SocialLDG
from Data.Dataset import get_datasets
from Data.DataLoader import get_dataloaders
from scripts import EarlyStopping, get_logits_y_true_pred, get_confidence_f1_accc, load_config
from Data.constants import device, original_subtasks, contact_classes, intention_classes, attitude_classes, \
    jpl_harper_action_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialLDG')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrained_encoder', type=str, required=True)
    return parser.parse_args()


def train_socialldg(args, config):
    MAX_EPOCH = 100
    subtasks = config['model']['subtasks']
    net = Encoder_SocialLDG(
        batch_size=config['train']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        encoder_gcn_layers=config['model']['encoder_gcn_layers'],
        encoder_time_layers=config['model']['encoder_time_layers'],
        keypoint_hidden_dim=config['model']['keypoint_hidden_dim'],
        num_heads=config['model']['num_heads'],
        representation_length=config['model']['representation_length'],
        dropout=config['train']['dropout'],
        hidden_dim=config['model']['hidden_dim'],
        task_token=config['model']['task_token'],
        n_heads=config['model']['n_heads'],
        subtasks=subtasks,
    )
    if args.pretrained_encoder:
        weights = torch.load(args.pretrained_encoder)
        weights = OrderedDict([[k, v.cuda(device)] for k, v in weights.items() if 'decoder' not in k])
        net.load_state_dict(weights, strict=False)
    net.to(device)

    if config['train']['epoch'] == 'early_stop':
        print("Early Stop is ON")
        early_stopper = EarlyStopping(minimize=False)
    else:
        print("Early Stop is OFF")
        early_stopper = None

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm']
    param_groups = [
        {"params": [], "lr": config['train']['encoder_learning_rate'], 'weight_decay': config['train']['weight_decay']},
        {"params": [], "lr": config['train']['SocialLDG_learning_rate'],
         'weight_decay': config['train']['weight_decay']},
        {"params": [], "lr": config['train']['encoder_learning_rate'], 'weight_decay': 0.0},
        {"params": [], "lr": config['train']['SocialLDG_learning_rate'], 'weight_decay': 0.0},
    ]
    for name, param in net.named_parameters():
        if 'classifier' in name:
            if any(nd in name for nd in no_decay):
                param_groups[3]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
        else:
            if any(nd in name for nd in no_decay):
                param_groups[2]['params'].append(param)
            else:
                param_groups[0]['params'].append(param)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
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
        trainset, valset, testset,
        sequence_length=config['data']['sequence_length'],
        batch_size=config['train']['batch_size'],
        zero_mask_rate=config['train']['zero_mask_rate'])
    for epoch in range(MAX_EPOCH):
        net.train()
        print('Training')
        con_cur_logits, con_cur_y_true, con_cur_y_pred, con_fut_logits, con_fut_y_true, con_fut_y_pred, int_logits, int_y_true, int_y_pred, att_logits, att_y_true, att_y_pred, act_cur_logits, act_cur_y_true, act_cur_y_pred, act_fut_logits, act_fut_y_true, act_fut_y_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        task_loss_sums = [0.0 for _ in range(len(subtasks))]
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for i, data in enumerate(train_loader):
            progress_bar.update(1)
            optimizer.zero_grad()
            inputs, (
                (con_cur_labels, con_fut_labels, int_labels, att_labels, act_cur_labels, act_fut_labels), _), _ = data
            con_cur_labels = con_cur_labels.to(device=device, dtype=torch.long)
            con_fut_labels = con_fut_labels.to(device=device, dtype=torch.long)
            int_labels = int_labels.to(device=device, dtype=torch.long)
            att_labels = att_labels.to(device=device, dtype=torch.long)
            act_cur_labels = act_cur_labels.to(device=device, dtype=torch.long)
            act_fut_labels = act_fut_labels.to(device=device, dtype=torch.long)
            outputs, _, edge_regularization = net(inputs)

            index_task = 0
            losses = []
            if 'contact_current' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], con_cur_labels))
                index_task += 1
            if 'contact_future' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], con_fut_labels))
                index_task += 1
            if 'intention' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], int_labels))
                index_task += 1
            if 'attitude' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], att_labels))
                index_task += 1
            if 'action_current' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], act_cur_labels))
                index_task += 1
            if 'action_future' in subtasks:
                losses.append(functional.cross_entropy(outputs[index_task], act_fut_labels))
                index_task += 1

            weights = [math.log(len(contact_classes)), math.log(len(contact_classes)), math.log(len(intention_classes)),
                       math.log(len(attitude_classes)), math.log(len(jpl_harper_action_classes)),
                       math.log(len(jpl_harper_action_classes))]
            total_loss = sum(
                [weights[original_subtasks.index(t)] * l for t, l in zip(subtasks, losses)])
            for i in range(len(subtasks)):
                task_loss_sums[i] += losses[i].item()

            total_loss += 1e-4 * edge_regularization

            total_loss.backward()
            optimizer.step()
            index_task = 0
            if 'contact_current' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_cur_labels)
                con_cur_logits.append(logits)
                con_cur_y_true += labels
                con_cur_y_pred += pred
                index_task += 1
            if 'contact_future' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_fut_labels)
                con_fut_logits.append(logits)
                con_fut_y_true += labels
                con_fut_y_pred += pred
                index_task += 1
            if 'intention' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], int_labels)
                int_logits.append(logits)
                int_y_true += labels
                int_y_pred += pred
                index_task += 1
            if 'attitude' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], att_labels)
                att_logits.append(logits)
                att_y_true += labels
                att_y_pred += pred
                index_task += 1
            if 'action_current' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_cur_labels)
                act_cur_logits.append(logits)
                act_cur_y_true += labels
                act_cur_y_pred += pred
                index_task += 1
            if 'action_future' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_fut_labels)
                act_fut_logits.append(logits)
                act_fut_y_true += labels
                act_fut_y_pred += pred
                index_task += 1
        progress_bar.close()
        result_str = 'training--> epoch: %d, ' % epoch
        total_f1, total_acc, total_confidence_score = 0, 0, 0
        if 'contact_current' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(con_cur_logits, con_cur_y_true, con_cur_y_pred)
            result_str += 'con_cur_confidence_score: %.2f, con_cur_acc: %.2f, con_cur_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'contact_future' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(con_fut_logits, con_fut_y_true, con_fut_y_pred)
            result_str += 'con_fut_confidence_score: %.2f, con_fut_acc: %.2f, con_fut_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'intention' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(int_logits, int_y_true, int_y_pred)
            result_str += 'int_confidence_score: %.2f, int_acc: %.2f, int_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'attitude' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(att_logits, att_y_true, att_y_pred)
            result_str += 'att_confidence_score: %.2f, att_acc: %.2f, att_f1: %.2f,' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'action_current' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(act_cur_logits, act_cur_y_true, act_cur_y_pred)
            result_str += 'act_cur_confidence_score: %.2f, act_cur_acc: %.2f%%, act_cur_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'action_future' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(act_fut_logits, act_fut_y_true, act_fut_y_pred)
            result_str += 'act_fut_confidence_score: %.2f, act_fut_acc: %.2f%%, act_fut_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        print(result_str + 'loss: %.4f, avg_confidence_score: %.2f, avg_f1: %.2f, avg_acc: %.2f' % (
            total_loss, total_confidence_score / len(subtasks), total_f1 * 100 / len(subtasks), total_acc * 100 / len(
                subtasks)))

        print('Validating')
        con_cur_logits, con_cur_y_true, con_cur_y_pred, con_fut_logits, con_fut_y_true, con_fut_y_pred, int_logits, int_y_true, int_y_pred, att_logits, att_y_true, att_y_pred, act_cur_logits, act_cur_y_true, act_cur_y_pred, act_fut_logits, act_fut_y_true, act_fut_y_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        net.eval()
        with (torch.no_grad()):
            for data in val_loader:
                inputs, (
                    (con_cur_labels, con_fut_labels, int_labels, att_labels, act_cur_labels, act_fut_labels),
                    _), _ = data
                con_cur_labels = con_cur_labels.to(device=device, dtype=torch.long)
                con_fut_labels = con_fut_labels.to(device=device, dtype=torch.long)
                int_labels = int_labels.to(device=device, dtype=torch.long)
                att_labels = att_labels.to(device=device, dtype=torch.long)
                act_cur_labels = act_cur_labels.to(device=device, dtype=torch.long)
                act_fut_labels = act_fut_labels.to(device=device, dtype=torch.long)
                outputs, _, _ = net(inputs)
                index_task = 0
                if 'contact_current' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_cur_labels)
                    con_cur_logits.append(logits)
                    con_cur_y_true += labels
                    con_cur_y_pred += pred
                    index_task += 1
                if 'contact_future' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_fut_labels)
                    con_fut_logits.append(logits)
                    con_fut_y_true += labels
                    con_fut_y_pred += pred
                    index_task += 1
                if 'intention' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], int_labels)
                    int_logits.append(logits)
                    int_y_true += labels
                    int_y_pred += pred
                    index_task += 1
                if 'attitude' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], att_labels)
                    att_logits.append(logits)
                    att_y_true += labels
                    att_y_pred += pred
                    index_task += 1
                if 'action_current' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_cur_labels)
                    act_cur_logits.append(logits)
                    act_cur_y_true += labels
                    act_cur_y_pred += pred
                    index_task += 1
                if 'action_future' in subtasks:
                    logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_fut_labels)
                    act_fut_logits.append(logits)
                    act_fut_y_true += labels
                    act_fut_y_pred += pred
                    index_task += 1
            result_str = 'validating--> epoch: %d, ' % epoch
            total_f1, total_acc, total_confidence_score = 0, 0, 0
            if 'contact_current' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(con_cur_logits, con_cur_y_true, con_cur_y_pred)
                result_str += 'con_cur_confidence_score: %.2f, con_cur_acc: %.2f, con_cur_f1: %.2f, ' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            if 'contact_future' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(con_fut_logits, con_fut_y_true, con_fut_y_pred)
                result_str += 'con_fut_confidence_score: %.2f, con_fut_acc: %.2f, con_fut_f1: %.2f, ' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            if 'intention' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(int_logits, int_y_true, int_y_pred)
                result_str += 'int_confidence_score: %.2f, int_acc: %.2f, int_f1: %.2f, ' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            if 'attitude' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(att_logits, att_y_true, att_y_pred)
                result_str += 'att_confidence_score: %.2f, att_acc: %.2f, att_f1: %.2f,' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            if 'action_current' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(act_cur_logits, act_cur_y_true, act_cur_y_pred)
                result_str += 'act_cur_confidence_score: %.2f, act_cur_acc: %.2f%%, act_cur_f1: %.2f, ' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            if 'action_future' in subtasks:
                confidence_score, acc, f1 = get_confidence_f1_accc(act_fut_logits, act_fut_y_true, act_fut_y_pred)
                result_str += 'act_fut_confidence_score: %.2f, act_fut_acc: %.2f%%, act_fut_f1: %.2f, ' % (
                    confidence_score.mean(), acc * 100, f1 * 100)
                total_confidence_score += confidence_score.mean()
                total_f1 += f1
                total_acc += acc
            print(result_str + 'avg_confidence_score: %.2f, avg_f1: %.2f, avg_acc: %.2f' % (
                total_confidence_score / len(subtasks), total_f1 * 100 / len(subtasks),
                total_acc * 100 / len(subtasks)))
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Warmup Phase: LR = {current_lr:.6f}")
        else:
            scheduler.step(total_f1 / len(subtasks))
        if config['train']['epoch'] == 'early_stop' and args.save_weights:
            if early_stopper.best_score < total_f1 / len(subtasks):
                model_name = 'encoder_SocialLDG_%s.pt' % '.'.join(subtasks)
                torch.save(net.state_dict(), 'new_weights/%s' % model_name)
            early_stopper(total_f1 / len(subtasks))
            print('The Best Score is %.2f, %d more epoch to go.' % (
                early_stopper.best_score * 100, early_stopper.patience - early_stopper.counter))
            if early_stopper.early_stop:
                print("Early stopping triggered, stopping training. The best score is %.2f" % (
                        early_stopper.best_score * 100))
                break
        elif epoch == config['train']['epochs'] - 1:
            model_name = 'encoder_SocialLDG_%s.pt' % '.'.join(subtasks)
            torch.save(net.state_dict(), 'new_weights/%s' % model_name)
            break
        else:
            print('------------------------------------------')

    print('Testing')
    con_cur_logits, con_cur_y_true, con_cur_y_pred, con_fut_logits, con_fut_y_true, con_fut_y_pred, int_logits, int_y_true, int_y_pred, att_logits, att_y_true, att_y_pred, act_cur_logits, act_cur_y_true, act_cur_y_pred, act_fut_logits, act_fut_y_true, act_fut_y_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    net.eval()
    with (torch.no_grad()):
        for data in test_loader:
            inputs, (
                (con_cur_labels, con_fut_labels, int_labels, att_labels, act_cur_labels, act_fut_labels),
                _), _ = data
            con_cur_labels = con_cur_labels.to(device=device, dtype=torch.long)
            con_fut_labels = con_fut_labels.to(device=device, dtype=torch.long)
            int_labels = int_labels.to(device=device, dtype=torch.long)
            att_labels = att_labels.to(device=device, dtype=torch.long)
            act_cur_labels = act_cur_labels.to(device=device, dtype=torch.long)
            act_fut_labels = act_fut_labels.to(device=device, dtype=torch.long)
            outputs, _, _ = net(inputs)
            index_task = 0
            if 'contact_current' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_cur_labels)
                con_cur_logits.append(logits)
                con_cur_y_true += labels
                con_cur_y_pred += pred
                index_task += 1
            if 'contact_future' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], con_fut_labels)
                con_fut_logits.append(logits)
                con_fut_y_true += labels
                con_fut_y_pred += pred
                index_task += 1
            if 'intention' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], int_labels)
                int_logits.append(logits)
                int_y_true += labels
                int_y_pred += pred
                index_task += 1
            if 'attitude' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], att_labels)
                att_logits.append(logits)
                att_y_true += labels
                att_y_pred += pred
                index_task += 1
            if 'action_current' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_cur_labels)
                act_cur_logits.append(logits)
                act_cur_y_true += labels
                act_cur_y_pred += pred
                index_task += 1
            if 'action_future' in subtasks:
                logits, pred, labels = get_logits_y_true_pred(outputs[index_task], act_fut_labels)
                act_fut_logits.append(logits)
                act_fut_y_true += labels
                act_fut_y_pred += pred
                index_task += 1
        result_str = 'testing--> epoch: %d, ' % epoch
        total_f1, total_acc, total_confidence_score = 0, 0, 0
        if 'contact_current' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(con_cur_logits, con_cur_y_true, con_cur_y_pred)
            result_str += 'con_cur_confidence_score: %.2f, con_cur_acc: %.2f, con_cur_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'contact_future' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(con_fut_logits, con_fut_y_true, con_fut_y_pred)
            result_str += 'con_fut_confidence_score: %.2f, con_fut_acc: %.2f, con_fut_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'intention' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(int_logits, int_y_true, int_y_pred)
            result_str += 'int_confidence_score: %.2f, int_acc: %.2f, int_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'attitude' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(att_logits, att_y_true, att_y_pred)
            result_str += 'att_confidence_score: %.2f, att_acc: %.2f, att_f1: %.2f,' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'action_current' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(act_cur_logits, act_cur_y_true, act_cur_y_pred)
            result_str += 'act_cur_confidence_score: %.2f, act_cur_acc: %.2f%%, act_cur_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        if 'action_future' in subtasks:
            confidence_score, acc, f1 = get_confidence_f1_accc(act_fut_logits, act_fut_y_true, act_fut_y_pred)
            result_str += 'act_fut_confidence_score: %.2f, act_fut_acc: %.2f%%, act_fut_f1: %.2f, ' % (
                confidence_score.mean(), acc * 100, f1 * 100)
            total_confidence_score += confidence_score.mean()
            total_f1 += f1
            total_acc += acc
        print(result_str + 'avg_confidence_score: %.2f, avg_f1: %.2f, avg_acc: %.2f' % (
            total_confidence_score / len(subtasks), total_f1 * 100 / len(subtasks),
            total_acc * 100 / len(subtasks)))
    return


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    train_socialldg(args, config)
