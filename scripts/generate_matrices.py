import os
import csv
from collections import OrderedDict
import torch
import torch.nn.functional as F
import argparse

from scripts import load_config
from Model import Encoder_SocialLDG
from Data.Dataset import get_datasets
from Data.DataLoader import SocialLDG_DataLoader
from Data.constants import device, jpl_harper_action_classes, contact_classes, intention_classes, attitude_classes, \
    intensity_classes, stage_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialLDG')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--mc_dropout', type=float, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrained_encoder_socialldg', type=str, required=True)
    parser.add_argument('--save_csv_path', type=str, required=True)
    return parser.parse_args()


def load_dataloader_and_model(pretrained_model_path, args, config):
    testset = get_datasets(data_path=args.data_path,
                           sequence_length=config['data']['sequence_length'],
                           future_length=config['data']['future_length'],
                           stride=config['data']['stride'],
                           test=True)
    test_loader = SocialLDG_DataLoader(dataset=testset,
                                       batch_size=config['train']['batch_size'],
                                       sequence_length=config['data']['sequence_length'],
                                       shuffle=False,
                                       drop_last=False,
                                       zero_mask_rate=config['train']['zero_mask_rate'])

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
        msg_pass_steps=config['model']['msg_pass_steps'],
        subtasks=config['model']['subtasks'],
        intermediate_supervision=config['model']['intermediate_supervision']
    )
    weights = torch.load(pretrained_model_path)
    weights = OrderedDict([[k, v.cuda(device)] for k, v in weights.items()])
    net.load_state_dict(weights, strict=False)
    net.to(device)
    return test_loader, net


def calc_batch_entropy(preds_list):
    stacked_preds = torch.stack(preds_list)
    mean_preds = stacked_preds.mean(dim=0)
    entropy = -torch.sum(mean_preds * torch.log(mean_preds + 1e-8), dim=-1)
    return entropy.cpu().tolist()


def generate_matrices(args, config):
    column_names = ["Sample_ID", "User_ID", "Stage_Label", "Intensity_Label",
                    "Label_Contact_Current", "Label_Contact_Future", 'Label_Intent', "Label_Attitude",
                    "Label_Action_Current", "Label_Action_Future",
                    # graph means incorporating information from other tasks
                    "Pred_Contact_Current_graph", "Pred_Contact_Future_graph", 'Pred_Intent_graph',
                    "Pred_Attitude_graph", "Pred_Action_Current_graph", "Pred_Action_Future_graph",
                    "Conf_Contact_Current_graph", "Conf_Contact_Future_graph", 'Conf_Intent_graph',
                    "Conf_Attitude_graph", "Conf_Action_Current_graph", "Conf_Action_Future_graph",
                    "Uncertainty_Entropy_Contact_Current_graph", "Uncertainty_Entropy_Contact_Future_graph",
                    'Uncertainty_Entropy_Intent_graph', "Uncertainty_Entropy_Attitude_graph",
                    "Uncertainty_Entropy_Action_Current_graph", "Uncertainty_Entropy_Action_Future_graph",

                    # graph intermediate means relying solely their own information before incorporating information from other tasks
                    "Pred_Contact_Current_graph_intermediate", "Pred_Contact_Future_graph_intermediate",
                    'Pred_Intent_graph_intermediate', "Pred_Attitude_graph_intermediate",
                    "Pred_Action_Current_graph_intermediate", "Pred_Action_Future_graph_intermediate",
                    "Conf_Contact_Current_graph_intermediate", "Conf_Contact_Future_graph_intermediate",
                    'Conf_Intent_graph_intermediate', "Conf_Attitude_graph_intermediate",
                    "Conf_Action_Current_graph_intermediate", "Conf_Action_Future_graph_intermediate",
                    "Uncertainty_Entropy_Contact_Current_graph_intermediate",
                    "Uncertainty_Entropy_Contact_Future_graph_intermediate",
                    'Uncertainty_Entropy_Intent_graph_intermediate', "Uncertainty_Entropy_Attitude_graph_intermediate",
                    "Uncertainty_Entropy_Action_Current_graph_intermediate",
                    "Uncertainty_Entropy_Action_Future_graph_intermediate",

                    "Weight_Contact_Current_to_Contact_Current", "Weight_Contact_Current_to_Contact_Future",
                    "Weight_Contact_Current_to_Intent", "Weight_Contact_Current_to_Attitude",
                    "Weight_Contact_Current_to_Action_Current", "Weight_Contact_Current_to_Action_Future",
                    "Weight_Contact_Future_to_Contact_Current", "Weight_Contact_Future_to_Contact_Future",
                    "Weight_Contact_Future_to_Intent", "Weight_Contact_Future_to_Attitude",
                    "Weight_Contact_Future_to_Action_Current", "Weight_Contact_Future_to_Action_Future",
                    "Weight_Intent_to_Contact_Current", "Weight_Intent_to_Contact_Future", "Weight_Intent_to_Intent",
                    "Weight_Intent_to_Attitude", "Weight_Intent_to_Action_Current", "Weight_Intent_to_Action_Future",
                    "Weight_Attitude_to_Contact_Current", "Weight_Attitude_to_Contact_Future",
                    "Weight_Attitude_to_Intent", "Weight_Attitude_to_Attitude", "Weight_Attitude_to_Action_Current",
                    "Weight_Attitude_to_Action_Future",
                    "Weight_Action_Current_to_Contact_Current", "Weight_Action_Current_to_Contact_Future",
                    "Weight_Action_Current_to_Intent", "Weight_Action_Current_to_Attitude",
                    "Weight_Action_Current_to_Action_Current", "Weight_Action_Current_to_Action_Future",
                    "Weight_Action_Future_to_Contact_Current", "Weight_Action_Future_to_Contact_Future",
                    "Weight_Action_Future_to_Intent", "Weight_Action_Future_to_Attitude",
                    "Weight_Action_Future_to_Action_Current", "Weight_Action_Future_to_Action_Future",
                    ]
    TASK_NAMES = ['Contact_Current', 'Contact_Future', 'Intent', 'Attitude', 'Action_Current', 'Action_Future']
    csv_dict = {}
    for cn in column_names:
        csv_dict[cn] = []
    test_loader, graph_net = load_dataloader_and_model(args.pretrained_encoder_socialldg, args, config)
    graph_net.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, (batch_labels, (batch_stage_labels, batch_video_labels, batch_user_labels)), _ = data
            (outputs, intermediate_outputs), (task_edge_index_list, edge_weights), _ = graph_net(inputs)
            con_cur_outputs, con_fut_outputs, int_outputs, att_outputs, act_cur_outputs, act_fut_outputs = outputs
            csv_dict["Sample_ID"] += batch_video_labels
            csv_dict["User_ID"] += batch_user_labels.cpu().tolist()
            csv_dict["Stage_Label"] += [stage_classes[int(i)] for i in batch_stage_labels.cpu().tolist()]
            intensity_label = torch.zeros_like(batch_labels[0])
            intensity_label[batch_labels[2] != 0] = 2
            intensity_label[batch_labels[2] == 0] = 1
            intensity_label[batch_labels[0] == 1] = 0
            csv_dict["Intensity_Label"] += [intensity_classes[int(i)] for i in intensity_label.cpu().tolist()]
            csv_dict["Label_Contact_Current"] += [contact_classes[int(i)] for i in batch_labels[0].cpu().tolist()]
            csv_dict["Label_Contact_Future"] += [contact_classes[int(i)] for i in batch_labels[1].cpu().tolist()]
            csv_dict["Label_Intent"] += [intention_classes[int(i)] for i in batch_labels[2].cpu().tolist()]
            csv_dict["Label_Attitude"] += [attitude_classes[int(i)] for i in batch_labels[3].cpu().tolist()]
            csv_dict["Label_Action_Current"] += [jpl_harper_action_classes[int(i)] for i in
                                                 batch_labels[4].cpu().tolist()]
            csv_dict["Label_Action_Future"] += [jpl_harper_action_classes[int(i)] for i in
                                                batch_labels[5].cpu().tolist()]
            confidence, pred = torch.softmax(con_cur_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Contact_Current_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Contact_Current_graph"] += [contact_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(con_fut_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Contact_Future_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Contact_Future_graph"] += [contact_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(int_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Intent_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Intent_graph"] += [intention_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(att_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Attitude_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Attitude_graph"] += [attitude_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(act_cur_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Action_Current_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Action_Current_graph"] += [jpl_harper_action_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(act_fut_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Action_Future_graph"] += confidence.cpu().tolist()
            csv_dict["Pred_Action_Future_graph"] += [jpl_harper_action_classes[int(i)] for i in pred.cpu().tolist()]

            con_cur_outputs, con_fut_outputs, int_outputs, att_outputs, act_cur_outputs, act_fut_outputs = intermediate_outputs
            confidence, pred = torch.softmax(con_cur_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Contact_Current_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Contact_Current_graph_intermediate"] += [contact_classes[int(i)] for i in
                                                                    pred.cpu().tolist()]
            confidence, pred = torch.softmax(con_fut_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Contact_Future_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Contact_Future_graph_intermediate"] += [contact_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(int_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Intent_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Intent_graph_intermediate"] += [intention_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(att_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Attitude_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Attitude_graph_intermediate"] += [attitude_classes[int(i)] for i in pred.cpu().tolist()]
            confidence, pred = torch.softmax(act_cur_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Action_Current_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Action_Current_graph_intermediate"] += [jpl_harper_action_classes[int(i)] for i in
                                                                   pred.cpu().tolist()]
            confidence, pred = torch.softmax(act_fut_outputs, dim=1).max(dim=1)
            csv_dict["Conf_Action_Future_graph_intermediate"] += confidence.cpu().tolist()
            csv_dict["Pred_Action_Future_graph_intermediate"] += [jpl_harper_action_classes[int(i)] for i in
                                                                  pred.cpu().tolist()]

            for n, task_n in enumerate(TASK_NAMES):
                for m, task_m in enumerate(TASK_NAMES):
                    csv_dict["Weight_%s_to_%s" % (task_n, task_m)] += edge_weights[:, m, n].cpu().tolist()

        con_cur_uncertainty_entropy, con_fut_uncertainty_entropy, int_uncertainty_entropy, att_uncertainty_entropy, act_cur_uncertainty_entropy, act_fut_uncertainty_entropy = [], [], [], [], [], []
        for data in test_loader:
            inputs, _, _ = data
            batch_con_cur, batch_con_fut, batch_int, batch_att, batch_act_cur, batch_act_fut = [], [], [], [], [], []
            for i in range(10):
                graph_net.dropout_rate = args.mc_dropout
                for m in graph_net.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                (outputs, _), _, _ = graph_net(inputs)
                con_cur_outputs, con_fut_outputs, int_outputs, att_outputs, act_cur_outputs, act_fut_outputs = outputs
                batch_con_cur.append(F.softmax(con_cur_outputs, dim=-1))
                batch_con_fut.append(F.softmax(con_fut_outputs, dim=-1))
                batch_int.append(F.softmax(int_outputs, dim=-1))
                batch_att.append(F.softmax(att_outputs, dim=-1))
                batch_act_cur.append(F.softmax(act_cur_outputs, dim=-1))
                batch_act_fut.append(F.softmax(act_fut_outputs, dim=-1))
            con_cur_uncertainty_entropy.extend(calc_batch_entropy(batch_con_cur))
            con_fut_uncertainty_entropy.extend(calc_batch_entropy(batch_con_fut))
            int_uncertainty_entropy.extend(calc_batch_entropy(batch_int))
            att_uncertainty_entropy.extend(calc_batch_entropy(batch_att))
            act_cur_uncertainty_entropy.extend(calc_batch_entropy(batch_act_cur))
            act_fut_uncertainty_entropy.extend(calc_batch_entropy(batch_act_fut))

        csv_dict['Uncertainty_Entropy_Contact_Current_graph'] = con_cur_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Contact_Future_graph'] = con_fut_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Intent_graph'] = int_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Attitude_graph'] = att_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Action_Current_graph'] = act_cur_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Action_Future_graph'] = act_fut_uncertainty_entropy

        con_cur_uncertainty_entropy, con_fut_uncertainty_entropy, int_uncertainty_entropy, att_uncertainty_entropy, act_cur_uncertainty_entropy, act_fut_uncertainty_entropy = [], [], [], [], [], []
        for data in test_loader:
            inputs, _, _ = data
            batch_con_cur, batch_con_fut, batch_int, batch_att, batch_act_cur, batch_act_fut = [], [], [], [], [], []
            for i in range(10):
                graph_net.dropout_rate = args.mc_dropout
                for m in graph_net.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                (_, intermediate_outputs), _, _ = graph_net(inputs)
                con_cur_outputs, con_fut_outputs, int_outputs, att_outputs, act_cur_outputs, act_fut_outputs = intermediate_outputs
                batch_con_cur.append(F.softmax(con_cur_outputs, dim=-1))
                batch_con_fut.append(F.softmax(con_fut_outputs, dim=-1))
                batch_int.append(F.softmax(int_outputs, dim=-1))
                batch_att.append(F.softmax(att_outputs, dim=-1))
                batch_act_cur.append(F.softmax(act_cur_outputs, dim=-1))
                batch_act_fut.append(F.softmax(act_fut_outputs, dim=-1))
            con_cur_uncertainty_entropy.extend(calc_batch_entropy(batch_con_cur))
            con_fut_uncertainty_entropy.extend(calc_batch_entropy(batch_con_fut))
            int_uncertainty_entropy.extend(calc_batch_entropy(batch_int))
            att_uncertainty_entropy.extend(calc_batch_entropy(batch_att))
            act_cur_uncertainty_entropy.extend(calc_batch_entropy(batch_act_cur))
            act_fut_uncertainty_entropy.extend(calc_batch_entropy(batch_act_fut))

        csv_dict['Uncertainty_Entropy_Contact_Current_graph_intermediate'] = con_cur_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Contact_Future_graph_intermediate'] = con_fut_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Intent_graph_intermediate'] = int_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Attitude_graph_intermediate'] = att_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Action_Current_graph_intermediate'] = act_cur_uncertainty_entropy
        csv_dict['Uncertainty_Entropy_Action_Future_graph_intermediate'] = act_fut_uncertainty_entropy

    csv_path = args.save_csv_path + 'new_matrices_uncertainty.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for i in range(len(csv_dict["Sample_ID"])):
            writer.writerow([csv_dict[key][i] for key in column_names])


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    generate_matrices(args, config)
