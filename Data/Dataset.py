from torch.utils.data import Dataset
import torch

import json
from Data import get_first_id, get_autoencoder_pose_tra_test_files
from Data.constants import coco_body_point_num, head_point_num, hands_point_num, jpl_user_num

action_dict = {'jpl_0': 0, 'jpl_1': 1, 'jpl_2': 2, 'jpl_3': 3, 'jpl_4': 4, 'jpl_5': 5, 'jpl_6': 7, 'jpl_7': 7,
               'jpl_8': 9, 'jpl_9': 11, 'harper_0': 10, 'harper_1': 7, 'harper_2': 11, 'harper_3': 2, 'harper_4': 6,
               'harper_5': 4, 'harper_6': 11}


class SocialLDGPoseDataset(Dataset):
    def __init__(self, data_files, sequence_length=10, future_length=10):
        super(SocialLDGPoseDataset, self).__init__()
        self.files = data_files
        self.sequence_length = sequence_length
        self.future_length = future_length
        self.frame_jump = sequence_length
        self.pose, self.labels = [], []
        for file in self.files:
            self.get_data_from_file(file)

    def get_data_from_file(self, file):
        is_jpl = True if 'JPL' in file else False
        with open(file, 'r') as f:
            feature_json = json.load(f)
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        frames = feature_json['frames']
        first_id = get_first_id(feature_json) if is_jpl else frames[0]['frame_id']
        if first_id == -1:
            return
        walk_frame, interact_start_frame, interact_end_frame = feature_json['walk_frame'], feature_json[
            'interact_start_frame'], feature_json['interact_end_frame']
        ori_keypoints_frames = torch.zeros(
            (frames[-1]['frame_id'] - first_id + 1, coco_body_point_num + head_point_num + hands_point_num, 3))
        for frame in frames:
            keypoints = torch.tensor(frame['keypoints'])
            ori_keypoints_frames[frame['frame_id'] - first_id, :, 0] = torch.clamp(keypoints[:, 0], min=0,
                                                                                   max=frame_width) / frame_width - 0.5
            ori_keypoints_frames[frame['frame_id'] - first_id, :, 1] = torch.clamp(keypoints[:, 1], min=0,
                                                                                   max=frame_height) / frame_height - 0.5
            ori_keypoints_frames[frame['frame_id'] - first_id, :, 2] = keypoints[:, 2]

        # Correcting differences in FPS in JPL and HARPER
        actual_seq_len = self.sequence_length * (3 if is_jpl else 1)
        actual_fut_len = self.future_length * (3 if is_jpl else 1)

        if ori_keypoints_frames.shape[0] < actual_seq_len / 2:
            return
        elif ori_keypoints_frames.shape[0] < actual_seq_len:
            last_frame = ori_keypoints_frames[-1:].expand(actual_seq_len - ori_keypoints_frames.shape[0], -1, -1)
            ori_keypoints_frames = torch.cat([ori_keypoints_frames, last_frame], dim=0)
        for frame_index in range(0, ori_keypoints_frames.shape[0] - actual_fut_len + 1,
                                 (3 if is_jpl else 1) * self.frame_jump):
            zero_frames_num = 0
            for i in range(frame_index, frame_index + actual_seq_len):
                if ori_keypoints_frames[i].all() == 0:
                    zero_frames_num += 1
            if zero_frames_num <= int(actual_seq_len * 0.5):
                filled_keypoints_frames = ori_keypoints_frames[frame_index:frame_index + actual_seq_len].clone()
                for i in range(frame_index, frame_index + actual_seq_len):
                    if (ori_keypoints_frames[i] == 0).all():
                        filled_keypoints_frames[i - frame_index] = filled_keypoints_frames[i - frame_index - 1]
                x_list = [filled_keypoints_frames[:, :coco_body_point_num],
                          filled_keypoints_frames[:, coco_body_point_num:head_point_num + coco_body_point_num],
                          filled_keypoints_frames[:, -hands_point_num:]]
                # Correcting differences in FPS in JPL and HARPER
                if is_jpl:
                    x_list = [x_list[0][::3], x_list[1][::3], x_list[2][::3]]

                contact_current_label = feature_json[
                    'contact_class'] if frame_index + first_id + actual_seq_len >= interact_start_frame and frame_index + first_id <= interact_end_frame else 0
                contact_future_label = feature_json[
                    'contact_class'] if frame_index + first_id + actual_seq_len + actual_fut_len >= interact_start_frame and frame_index + first_id + actual_seq_len <= interact_end_frame else 0
                if feature_json['intention_class'] == 2:
                    intention_label = 2
                else:
                    if frame_index + first_id > interact_end_frame:
                        intention_label = 2
                    else:
                        intention_label = feature_json['intention_class']
                attitude_label = feature_json[
                    'attitude_class'] if walk_frame <= frame_index + first_id + actual_seq_len and frame_index + first_id <= interact_end_frame else 2
                if frame_index + first_id + actual_seq_len < walk_frame or frame_index + first_id > interact_end_frame:
                    action_current_label = 11  # Approaching
                elif walk_frame <= frame_index + first_id + actual_seq_len < interact_start_frame:
                    action_current_label = 8  # Not Interacting
                else:
                    action_current_label = action_dict[
                        '%s_%d' % ('jpl' if is_jpl else 'harper', feature_json['action_class'])]
                if frame_index + first_id + actual_seq_len + actual_fut_len < walk_frame or frame_index + first_id + actual_seq_len > interact_end_frame:
                    action_future_label = 11  # Approaching
                elif walk_frame <= frame_index + first_id + actual_seq_len + actual_fut_len < interact_start_frame:
                    action_future_label = 8  # Not Interacting
                else:
                    action_future_label = action_dict[
                        '%s_%d' % ('jpl' if is_jpl else 'harper', feature_json['action_class'])]

                if feature_json['intention_class'] == 0:
                    if frame_index + first_id < walk_frame:
                        interaction_stage_label = 0  # Pre-Interaction
                    elif frame_index + first_id + actual_seq_len < interact_start_frame:
                        interaction_stage_label = 1  # Approaching
                    elif frame_index + first_id <= interact_end_frame:
                        interaction_stage_label = 2  # Interacting
                    elif frame_index + first_id > interact_end_frame:
                        interaction_stage_label = 3  # Post-Interaction
                else:
                    interaction_stage_label = 4  # No-Interaction

                video_label = '%s_%d' % (file, frame_index + first_id)
                labels = (
                    contact_current_label, contact_future_label, intention_label, attitude_label, action_current_label,
                    action_future_label), (interaction_stage_label, video_label)
                self.pose.append(x_list)
                self.labels.append(labels)

    def __getitem__(self, idx):
        return self.pose[idx], self.labels[idx]

    def __len__(self):
        return len(self.pose)


def get_datasets(data_path, sequence_length=10, future_length=10, test=False):
    print('Loading data from JPL and HARPER dataset for AutoEncoder')
    tra_files, val_files, test_files = get_autoencoder_pose_tra_test_files(data_path=data_path)
    result_str = ''
    if not test:
        trainset = SocialLDGPoseDataset(data_files=tra_files, sequence_length=sequence_length,
                                        future_length=future_length)
        result_str += 'Train_set_size: %d, ' % len(trainset)
        valset = SocialLDGPoseDataset(data_files=val_files, sequence_length=sequence_length,
                                      future_length=future_length)
        result_str += 'Validation_set_size: %d, ' % len(valset)
    testset = SocialLDGPoseDataset(data_files=test_files, sequence_length=sequence_length, future_length=future_length)
    result_str += 'Test_set_size: %d, ' % len(testset)
    print(result_str)
    if not test:
        return trainset, valset, testset
    else:
        return testset
