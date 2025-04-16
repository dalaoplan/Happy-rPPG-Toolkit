import torch
import torch.nn as nn
import torch.nn.functional as F
# tr = torch
import torch.fft


def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    targets = targets[:, :]
    # predictions = torch.squeeze(predictions)
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions, dim=-1, keepdim=True)) / torch.std(predictions, dim=-1,
                                                                                            keepdim=True)
    targets = (targets - torch.mean(targets, dim=-1, keepdim=True)) / torch.std(targets, dim=-1, keepdim=True)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1] if len(predictions.shape) > 1 else 1
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)
        if len(targets.shape) == 1:
            targets = targets.view(1, -1)
        return neg_Pearson_Loss(predictions, targets)


class PSDKLLoss(nn.Module):
    def __init__(self, Fs, high_pass=45, low_pass=150, weight_factor=2.0, kl_weight=0.5, alpha=50, loss_scale=1):
        """
        结合 MSE 和 KL Loss 计算 PSD 误差。

        :param Fs: 采样率 (Hz)
        :param high_pass: 最低心率 (BPM)
        :param low_pass: 最高心率 (BPM)
        :param weight_factor: 心率频段的权重
        :param kl_weight: KL 损失的权重
        :param alpha: 频率滤波平滑因子
        :param loss_scale: 损失值放大系数，防止损失过小
        """
        super(PSDKLLoss, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.weight_factor = weight_factor  # 心率频段加权
        self.kl_weight = kl_weight  # KL Loss 的权重
        self.alpha = alpha  # 控制平滑筛选
        self.loss_scale = loss_scale  # 防止损失过小
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def compute_psd(self, x, zero_pad=0):
        """ 计算并归一化 PSD """
        x = x - torch.mean(x, dim=-1, keepdim=True)  # 去均值
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant', 0)

        # 计算 PSD
        x = torch.fft.rfft(x, dim=-1, norm='forward')
        x = torch.real(x) ** 2 + torch.imag(x) ** 2

        # 频率筛选
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[-1], device=x.device)
        weight = torch.sigmoid(self.alpha * (freqs - self.high_pass / 60)) * torch.sigmoid(-self.alpha * (freqs - self.low_pass / 60))
        x = x * weight

        # **避免数值过小**，放缩 PSD
        x = x / (1e-8 + torch.max(x, dim=-1, keepdim=True)[0])

        # 归一化
        x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-8)
        return x

    def forward(self, pred, target, zero_pad=0):
        """
        计算 PSD 误差 (MSE + KL Loss)
        :param pred: 预测的时域信号 (batch_size, time_steps)
        :param target: 真实时域信号 (batch_size, time_steps)
        :return: 组合损失
        """
        # 计算 PSD
        pred_psd = self.compute_psd(pred, zero_pad)
        target_psd = self.compute_psd(target, zero_pad)

        # 计算 Nyquist 频率
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, pred_psd.shape[-1], device=pred_psd.device)

        # 选择心率相关频率范围
        valid_mask = (freqs >= self.high_pass / 60) & (freqs <= self.low_pass / 60)

        # 计算权重 (提高心率频率区间的重要性)
        weights = 1 + self.weight_factor * valid_mask.to(pred_psd.dtype)

        # 计算 MSE Loss
        mse_loss = self.mse(pred_psd * weights, target_psd * weights)

        # **优化 KL Loss 计算**
        kl_loss = F.kl_div((pred_psd + 1e-8).log(), target_psd, reduction="batchmean")

        # 组合损失
        loss = (mse_loss + self.kl_weight * kl_loss) * self.loss_scale  # **放大损失**

        return loss


class MyLoss(nn.Module):
    def __init__(self, fs=30):
        super(MyLoss, self).__init__()
        self.neg_Pearson_Loss = NegPearsonLoss()
        self.PSDKLLoss = PSDKLLoss(fs)

    def forward(self, pred, target, epoch, fs):
        loss_neg = self.neg_Pearson_Loss(pred, target)  # 约 1.0
        loss_psd = self.PSDKLLoss(pred, target)         # 约 0.5

        # 处理 NaN
        if torch.isnan(loss_neg):
            loss_neg = torch.where(torch.isnan(loss_neg), torch.tensor(0.0, device=loss_neg.device), loss_neg)

        # 训练前 15 轮：loss_neg 占主导，训练后 15 轮：loss_psd 占主导
        # if epoch <= 15:
        #     a = 5.0 - 4.0 * (epoch / 15.0)  # a: 5.0 → 1.0
        #     b = 1.0 + 4.0 * (epoch / 15.0)  # b: 1.0 → 5.0
        # else:
        #     a = 1.0
        #     b = 5.0
        a = 0.2
        b = 1

        # 计算最终损失
        loss = a * loss_neg + b * loss_psd
        return loss       #, loss_neg, loss_psd


if __name__ == '__main__':
    # torch.manual_seed(42)

    # 生成伪造数据
    batch_size = 4
    seq_length = 100
    pred = torch.rand(batch_size, seq_length)
    target = torch.rand(batch_size, seq_length)

    loss_fn = MyLoss(fs=30)

    loss, loss_neg, loss_psd = loss_fn(pred, target, 1)
    print(f"loss type:{type(loss)}, loss_neg type:{type(loss_neg)}, loss_psd{type(loss_psd)}")