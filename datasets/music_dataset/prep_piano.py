"""
# Dataset preparation

### h5 file generation

- unzip "piano_dataset.zip" in wav_piano
- run prep_piano.py from /music_dataset/wav_piano/ to create a h5 container with labels and train files of the entire dataset containing all 32 Sonatas.

```
python ../prep_piano.py \
--file-list piano_list.txt \
--scale 4 \
--sr 16000 \
--dimension 8192 \
--stride 8192 \
```

- notes: The dimension parameter and sampling rate define the absolute length of a patch (dim/sr= length patch)

"""

""" Script to create piano Dataset, Adopted from https://github.com/kuleshov/audio-super-res"""

import numpy as np
import os, argparse
import librosa
import soundfile as sf
from scipy import interpolate
from scipy.signal import decimate
from typing import Dict
from shutil import copyfile, move


def create_patches(d, s, sr, scale, file_list, num_files):
    """
    Creates high and low res patches based on configuration of all wav input files.
    Files are placed in 'raw_split' folder.

    :param d: patch dimension
    :param s: patch stride
    :param sr: sampling rate
    :param scale: scaling factor
    :param file_list: list of wav files
    :param num_files: number of files to be processed
    """
    os.makedirs('raw_split', exist_ok=True)

    for j, file_path in enumerate(file_list):
        if j % 1 == 0: print('%d/%d' % (j, num_files))

        # load audio file
        x, fs = librosa.load(file_path, sr=sr)

        # crop so that it works with scaling ratio
        x_len = len(x)
        x = x[: x_len - (x_len % scale)]

        # generate low-res version
        x_lr = decimate(x, scale)
        x_lr = upsample(x_lr, scale)
        assert len(x) % scale == 0
        assert len(x_lr) == len(x)

        # generate patches
        max_i = len(x) - d + 1
        for i in range(0, max_i, s):
            # keep only a fraction of all the patches
            hr_patch = np.array(x[i: i + d])
            lr_patch = np.array(x_lr[i: i + d])
            assert len(hr_patch) == d
            assert len(lr_patch) == d
            sf.write('./raw_split/' + 'patch_hr_' + str(i) + '.wav', hr_patch, sr)
            sf.write('./raw_split/' + 'patch_lr_' + str(i) + '.wav', lr_patch, sr)


def split_dataset(args, split_ratio: Dict[str, float] = {"train": 0.88, "val": 0.06, "test": 0.06}):

    """
    Splits dataset into train, val and test datasets

    :param args: Arguments defining low res sample generation and os paths
    :param split_ratio: Defines split ratio of dataset between train, val and test files
    """

    # Make a list of all files to be processed
    file_list = []
    file_extensions = set(['.wav'])
    with open(args.file_list) as f:
        for line in f:
            filename = line.strip()
            ext = os.path.splitext(filename)[1]
            if ext in file_extensions:
                file_list.append(os.path.join('wav_piano', filename))

    num_files = len(file_list)
    print(len(file_list))

    # Create raw patches:
    create_patches(args.dimension,args.stride,args.sr,args.scale,file_list,num_files)

    # list all patches in raw_split:
    hr_patches, lr_patches = list(), list()
    filenames = os.listdir('./raw_split')
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.wav':
            filenumber = filename.split('_')[-1][:-len('.wav')]
            if filename.split('_')[1] == 'lr':
                lr_patches.append('patch_lr_' + filenumber)
            elif filename.split('_')[1] == 'hr':
                hr_patches.append('patch_hr_' + filenumber)

    # split in 88-6-6:
    all_patches = np.arange(len(hr_patches))
    # randomly pick patches:
    np.random.shuffle(all_patches)
    patches_to_split = np.array(all_patches)
    print('Patches to split: ' + str(patches_to_split.shape[0]))

    # Split into train, val and test
    # create directories if necessary:
    os.makedirs('train', exist_ok=True)
    os.makedirs('val', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    start_idx, last_idx = 0, 0
    for k, v in split_ratio.items():
        num_wavs = int(np.floor(v * len(patches_to_split)))
        last_idx += num_wavs
        # move files:
        for i in range(start_idx,last_idx):
            # high res
            src = os.path.join('./raw_split', hr_patches[patches_to_split[i]] + '.wav')
            dst = os.path.join('./',k, hr_patches[patches_to_split[i]] + '.wav')
            copyfile(src, dst)
            # low res:
            src = os.path.join('./raw_split', lr_patches[patches_to_split[i]] + '.wav')
            dst = os.path.join('./', k, lr_patches[patches_to_split[i]] + '.wav')
            copyfile(src, dst)
        start_idx += num_wavs

    print('Done!')

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-list', default='./wav_piano/piano_list.txt',
                        help='list of input wav files to process')
    parser.add_argument('--scale', type=int, default=4,
                        help='scaling factor')
    parser.add_argument('--dimension', type=int, default=8192,
                        help='dimension of patches--use -1 for no patching')
    parser.add_argument('--stride', type=int, default=8192,
                        help='stride when extracting patches')
    parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')

    arguments = parser.parse_args()
    split_dataset(args=arguments)
