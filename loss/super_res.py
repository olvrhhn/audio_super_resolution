import torch

class MSE(torch.nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

        self.criterion = torch.nn.MSELoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        :param pred: super res patch
        :param gt: ground truth patch
        :return: L2 Loss
        """
        return self.criterion(pred, gt)
