import torch
import argparse
from collections import OrderedDict
from Model import Encoder_SocialLDG
from Data.Dataset import get_datasets
from Data.DataLoader import SocialLDG_DataLoader
from scripts import load_config,get_logits_y_true_pred,get_confidence_f1_accc
from Data.constants import device


def parse_args():
    parser = argparse.ArgumentParser(description='Test SocialLDG')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    return parser.parse_args()


def test_socialldg(args, config):
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
    weights = torch.load(args.checkpoint_path)
    weights = OrderedDict([[k, v.cuda(device)] for k, v in weights.items()])
    net.load_state_dict(weights, strict=True)
    net.to(device)
    net.eval()
    testset = get_datasets(
        data_path=args.data_path,
        sequence_length=config['data']['sequence_length'],
        future_length=config['data']['future_length'],
        test=True)
    test_loader = SocialLDG_DataLoader(
        dataset=testset,
        batch_size=config['train']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        shuffle=False,
        drop_last=False,
        zero_mask_rate=0)
    print('Testing')
    con_cur_logits, con_cur_y_true, con_cur_y_pred, con_fut_logits, con_fut_y_true, con_fut_y_pred, int_logits, int_y_true, int_y_pred, att_logits, att_y_true, att_y_pred, act_cur_logits, act_cur_y_true, act_cur_y_pred, act_fut_logits, act_fut_y_true, act_fut_y_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
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
        result_str = 'testing--> '
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
    test_socialldg(args, config)
