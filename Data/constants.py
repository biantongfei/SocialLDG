import os
import torch
import numpy as np
import random


def set_seed():
    seed = 42
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)  # Python 随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器
    torch.cuda.manual_seed_all(seed)  # 多 GPU 训练时，所有 GPU 固定种子
    # 关键设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark=True 会让 cuDNN 在启动时寻找最快算法，这在不同硬件上肯定不同


set_seed()

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device("cuda:0")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
else:
    print('Using CPU')
    device = torch.device('cpu')
dtype = torch.float32
intention_classes = ['Interacting', 'Interested', 'Not_Interested']
attitude_classes = ['Positive', 'Negative', 'Not_Interacting']
jpl_harper_action_classes = ['Handshake', 'Hug', 'Pet', 'Wave', 'Punch', 'Throw', 'Kick', 'Gaze', 'Walk', 'Leave',
                             'Crash', 'No_Response']
contact_classes = ['No', 'Yes']

coco_body_point_num = 23
head_point_num = 68
hands_point_num = 42

coco_body_l_pair = [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
                    [5, 7], [7, 9], [6, 8], [8, 10],  # Body
                    [5, 6], [11, 12], [5, 11], [6, 12],
                    [0, 5], [0, 6],
                    [11, 13], [12, 14], [13, 15], [14, 16],
                    [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]]
head_l_pair = [[23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
               [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39],  # jawline
               [40, 41], [41, 42], [42, 43], [43, 44], [44, 50],  # right eyebrow
               [45, 46], [46, 47], [47, 48], [48, 49], [45, 50],  # left eyebrow
               [50, 51], [51, 52], [52, 53], [53, 56], [54, 55], [55, 56], [56, 57], [57, 58],  # nose
               [59, 60], [60, 61], [61, 62], [62, 63], [63, 64], [64, 59],  # right eye
               [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 65],  # left eye
               [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81],
               [81, 82], [82, 71], [71, 83], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [87, 77],
               [87, 88], [88, 89], [89, 90], [90, 83]]
hand_l_pair = [[91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
               [100, 101], [101, 102],
               [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
               [110, 111], [112, 113],
               [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
               [121, 122], [122, 123],
               [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
               [131, 132]]

coco_wholebody_l_pair = coco_body_l_pair + head_l_pair + hand_l_pair + [[52, 0], [0, 56], [0, 53], [91, 9],
                                                                        [92, 9], [96, 9], [100, 9], [104, 9],
                                                                        [108, 9], [112, 10], [113, 10], [117, 10],
                                                                        [121, 10], [125, 10], [129, 10]]

jpl_user_num = 15
harper_user_num = 17
original_subtasks = ['contact_current', 'contact_future', 'intention', 'attitude', 'action_current', 'action_future']

if __name__ == '__main__':
    print(len(coco_wholebody_l_pair))
