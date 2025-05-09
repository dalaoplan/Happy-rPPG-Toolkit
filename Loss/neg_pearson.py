import torch
import torch.nn as nn

tr = torch
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

    def forward(self, predictions, targets, epoch, fs):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)
        if len(targets.shape) == 1:
            targets = targets.view(1, -1)
        return neg_Pearson_Loss(predictions, targets)


if __name__ == '__main__':
    input = torch.randn(3, 160)
    target = torch.randn((3, 158))
    net = NegPearsonLoss()
    out = net(input, target, 1, 30)

    print(out)
