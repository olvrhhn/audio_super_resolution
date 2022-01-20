import numpy as np
import torch
#import auraloss
import librosa


class averageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reduce_mean(self, x, axis):
        sorted(axis)
        axis = list(reversed(axis))
        for d in axis:
            x = torch.mean(x, dim=d)
        return x

    def snr(self, y, y_pred):
        # snr = auraloss.time.SNRLoss()
        # test = snr(torch.from_numpy(y_pred), torch.from_numpy(y))

        sqrt_l2_loss = np.sqrt(np.mean((y_pred - y) ** 2 + 1e-6, axis=(1, 2)))
        sqrn_l2_norm = np.sqrt(np.mean(y ** 2, axis=(1, 2)))
        snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
        avg_snr = np.mean(snr, axis=0)
        self.val_snr = avg_snr


    def lsd(self, input1, input2):
        win_length = input1.shape[2]
        x = torch.squeeze(input1)
        x = torch.stft(x, win_length, win_length)
        x = torch.log(torch.abs(x) ** 2 + 1e-8)
        x_hat = torch.squeeze(input2)
        x_hat = torch.stft(x_hat, win_length, win_length)
        x_hat = torch.log(torch.abs(x_hat) ** 2 + 1e-8)
        lsd = self.reduce_mean(torch.sqrt(self.reduce_mean(torch.mul(x - x_hat, x - x_hat), axis=[1, 2])) + 1e-8, axis=[0])
        self.val_lsd = lsd


    def get_power(self, x):
        S = librosa.stft(x.squeeze(0).squeeze(0), 2048)
        S = np.log(np.abs(S)**2 + 1e-8)
        return S


    def compute_log_distortion(self, x_hr, x_pr):
        S1 = self.get_power(x_hr)
        S2 = self.get_power(x_pr)
        lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis=0)
        self.val_lsd = min(lsd, 10.)


    def reset(self):
        self.val_snr = 0
        self.avg_snr = 0
        self.sum_snr = 0

        self.count = 0

        self.val_lsd = 0
        self.avg_lsd = 0
        self.sum_lsd = 0


    def update(self,
               gt,
               out,
               n: int = 1):

        self.snr(gt, out)
        self.compute_log_distortion(gt, out)

        self.count += n
        self.sum_snr += self.val_snr * n
        self.avg_snr = self.sum_snr / self.count
        self.sum_lsd += self.val_lsd * n
        self.avg_lsd = self.sum_lsd / self.count


    def get_score(self):
        return self.avg_snr, self.avg_lsd