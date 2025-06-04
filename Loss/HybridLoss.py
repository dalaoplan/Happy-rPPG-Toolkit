import torch
import torch.nn as nn
import torch.nn.functional as F



class HybridLoss(nn.Module):
    def __init__(self, fs, alpha=1, beta=0.5, low_cutoff=0.7, high_cutoff=2.7):
        super(HybridLoss, self).__init__()
        self.fs = fs
        self.alpha = alpha  # 带通损失权重
        self.beta = beta    # Pearson损失权重
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def forward(self, pred, gt):
        band_limited_loss = self.band_limited_spectral_loss(pred, gt)
        neg_pearson_loss = self.neg_pearson_loss(pred, gt)

        total_loss = self.alpha * band_limited_loss + self.beta * neg_pearson_loss
        return total_loss

    def band_limited_spectral_loss(self, pred, gt):
        """频域带通损失"""
        pred_fft = torch.fft.rfft(pred, norm='forward')
        pred_fft = torch.real(pred_fft) ** 2 + torch.imag(pred_fft) ** 2

        gt_fft = torch.fft.rfft(gt, norm='forward')
        gt_fft = torch.real(gt_fft) ** 2 + torch.imag(gt_fft) ** 2


        freqs = torch.fft.rfftfreq(pred.shape[-1], d=1 / self.fs).to(pred.device)

        mask = (freqs >= self.low_cutoff) & (freqs <= self.high_cutoff)
        pred_energy = pred_fft[:, mask].sum(dim=-1)
        gt_energy = gt_fft[:, mask].sum(dim=-1)

        loss = F.l1_loss(pred_energy, gt_energy)
        return loss

    def neg_pearson_loss(self, preds, labels):
        """负皮尔逊相关损失"""
        loss = 0
        for i in range(preds.shape[0]):
            x = preds[i]
            y = labels[i]
            sum_x = torch.sum(x)
            sum_y = torch.sum(y)
            sum_xy = torch.sum(x * y)
            sum_x2 = torch.sum(x * x)
            sum_y2 = torch.sum(y * y)
            N = preds.shape[1]
            numerator = N * sum_xy - sum_x * sum_y
            denominator = torch.sqrt((N * sum_x2 - sum_x ** 2) * (N * sum_y2 - sum_y ** 2) + 1e-8)
            pearson = numerator / (denominator + 1e-8)
            loss += 1 - pearson
        return loss / preds.shape[0]

