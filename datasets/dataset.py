import os, torch
from typing import Tuple
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from scipy import interpolate
from scipy.signal import decimate
import h5py
import soundfile as sf

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp


class Sonata32Dataset(Dataset):

    def __init__(self,
                 root: str,
                 target_type: str,
                 sr: int,
                 scale: int,
                 dimension: int,
                 stride: int):

        # Set up paths to data:
        self.target_type = target_type
        self.root = os.path.join(root, target_type)
        # load numpy file:
        data_raw = np.load(root + '/music_' + target_type + '.npy')
        # set sampling rate:
        self.sr = sr

        # change dtype to float32:
        data_raw_32 = data_raw.astype(np.float32)
        # reshape input to one long patch:
        data_long = data_raw_32.reshape(-1)

        # low_res data generation:

        # crop so that it works with scaling ratio
        data_long_len = len(data_long)
        data_long = data_long[: data_long_len - (data_long_len % scale)]

        # generate low-res version
        data_lr = decimate(data_long, scale)
        data_lr = upsample(data_lr, scale)
        assert len(data_long) % scale == 0
        assert len(data_lr) == len(data_long)

        # generate patches
        self.data_list, self.gt_list = [], []
        max_i = len(data_long) - dimension + 1
        for i in range(0, max_i, stride):
            # keep only a fraction of all the patches
            hr_patch = np.array(data_long[i: i + dimension])
            lr_patch = np.array(data_lr[i: i + dimension])
            assert len(hr_patch) == dimension
            assert len(lr_patch) == dimension
            self.data_list.append(lr_patch)
            self.gt_list.append(hr_patch)

        # sf.write('low_res_patch.wav',self.data_list[31],16000)
        # sf.write('high_res_patch.wav', self.gt_list[31], 16000)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:

        # transform to tensor:
        data = torch.tensor(self.data_list[idx])
        gt = torch.tensor(self.gt_list[idx])

        return (torch.unsqueeze(data, 0), self.sr, torch.unsqueeze(gt, 0), self.sr)


class SingleSpeaker(Dataset):

    def __init__(self,
                 root: str,
                 sr:int = 16000):

        # Set up paths to data:
        # Set sampling rate:
        self.sr = sr
        # load h5 file:
        data, gt = self.load_h5(root)
        # transform to tensor
        self.data = torch.tensor(data)
        self.gt = torch.tensor(gt)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:

        data = self.data[idx, :, 0]
        gt = self.gt[idx, :, 0]

        return (torch.unsqueeze(data, 0), self.sr, torch.unsqueeze(gt, 0), self.sr)


    @staticmethod
    def load_h5(h5_path):
        with h5py.File(h5_path, 'r') as hf:
            # print('List of arrays in input file:', list(hf.keys()))
            X = np.array(hf.get('data'))
            Y = np.array(hf.get('label'))
            # print('Shape of X:', X.shape)
            # print('Shape of Y:', Y.shape)

        return X, Y


class SingleSpeakerWav(Dataset):

    def __init__(self,
                 root: str,
                 target_type: str):

        # Set up paths to data and groundtruths
        self.target_type = target_type
        self.root = os.path.join(root, target_type)
        self.data_list, self.gt_list = [], []
        for file_path in os.listdir(self.root):
            if file_path.startswith("patch_hr"):
                self.gt_list.append(os.path.join(root,target_type, file_path))
            elif file_path.startswith("patch_lr"):
                self.data_list.append(os.path.join(root,target_type, file_path))
        # Test if # gt and data match:
        assert len(self.gt_list) == len(self.data_list), "Number of high res and ground truth patches does not match!"

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:

        # Read image, groundtruth segmentation and convert label to index
        data_wav, data_sr = torchaudio.load(self.data_list[idx])
        gt_wav, gt_sr = torchaudio.load(self.gt_list[idx])

        return (data_wav, data_sr, gt_wav, gt_sr)
