import numpy as np
import yaml
import torch
from torch.nn import functional
from sklearn.metrics import f1_score


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001, minimize=True):
        """
        Args:
            patience (int): 在上次验证集损失改善后，等待多少个 epoch 再停止。
                            (默认: 10)
            verbose (bool): 如果为 True，则打印有关早停的消息。
                            (默认: True)
            delta (float):  被认为是“实质性”改善的最小变化量。
                            (默认: 0.0001)
            path (str):    保存最佳模型检查点的路径。
                            (默认: 'best_model_checkpoint.pt')
        """
        self.patience = patience
        self.delta = delta
        self.minimize = minimize

        # 跟踪器
        self.epoch = 0
        self.counter = 0  # 跟踪没有改善的 epoch 数量
        self.best_score = np.inf if minimize else -np.inf  # 跟踪最佳的验证集损失 (越低越好)
        self.early_stop = False  # 停止标志

    def __call__(self, val_result):
        """
        在每个 epoch 验证后调用此方法

        Args:
            val_loss (float): 当前 epoch 的验证集损失
            model (torch.nn.Module): 正在训练的模型
        """

        # 检查当前损失是否“实质性”地优于最佳分数
        if (self.minimize and val_result < self.best_score - self.delta) or (
                not self.minimize and val_result > self.best_score + self.delta):
            # --- 1. 损失改善 ---
            # 更新最佳分数并重置计数器
            self.best_score = val_result
            self.counter = 0

        else:
            # --- 2. 损失没有改善 ---
            self.counter += 1

            # 检查是否达到了耐心阈值
            if self.counter >= self.patience:
                self.early_stop = True


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def get_logits_y_true_pred(outputs, labels):
    logits = outputs.squeeze(-1).detach()
    outputs = torch.softmax(outputs, dim=1)
    _, pred = torch.max(outputs, dim=1)
    return logits, pred.tolist(), labels.tolist()


def get_confidence_f1_accc(logits, y_true, y_pred):
    logits = torch.cat(logits, dim=0)
    confidence_score = torch.max(functional.softmax(logits, dim=1), dim=1)[0]
    y_true, y_pred = torch.Tensor(y_true), torch.Tensor(y_pred)
    acc = y_pred.eq(y_true).sum().float().item() / y_pred.size(dim=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return confidence_score, acc, f1
