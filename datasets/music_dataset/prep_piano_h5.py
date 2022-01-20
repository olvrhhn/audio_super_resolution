"""
Create an HDF5 file of patches for training super-resolution model.
adapted from https://github.com/kuleshov/audio-super-res
"""

import os, argparse
import numpy as np
import h5py
import pickle

import librosa
from scipy import interpolate
from scipy.signal import decimate

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--file-list',
                    help='list of input wav files to process')
parser.add_argument('--out',
                    default=os.path.join(os.getcwd(), "datasets", "vctk"),
                    help='path to output h5 archive')
parser.add_argument('--scale', type=int, default=2,
                    help='scaling factor')
parser.add_argument('--dimension', type=int,
                    help='dimension of patches--use -1 for no patching')
parser.add_argument('--stride', type=int, default=3200,
                    help='stride when extracting patches')
parser.add_argument('--interpolate', action='store_true',
                    help='interpolate low-res patches with cubic splines')
parser.add_argument('--low-pass', action='store_true',
                    help='apply low-pass filter when generating low-res patches')
parser.add_argument('--batch-size', type=int, default=128,
                    help='we produce # of patches that is a multiple of batch size')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')
parser.add_argument('--sam', type=float, default=1.,
                    help='subsampling factor for the data')
parser.add_argument('--full_sample', type=bool, default=True);

args = parser.parse_args()

# ----------------------------------------------------------------------------

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def add_data(h5_file, inputfiles, args, save_examples=False):
    # load numpy file:
    data_raw = np.load(inputfiles)
    # change dtype to float32:
    data_raw_32 = data_raw.astype(np.float32)
    # # reshape input to one long patch:
    # x = data_raw_32.reshape(-1)
    # patches to extract and their size
    if args.dimension is not -1:
        if args.interpolate:
            d, d_lr = args.dimension, args.dimension
            s, s_lr = args.stride, args.stride
        else:
            d, d_lr = args.dimension, args.dimension / args.scale
            s, s_lr = args.stride, args.stride / args.scale
    hr_patches, lr_patches = list(), list()

    # crop so that it works with scaling ratio
    num_files = data_raw_32.shape[0]
    for j in range(num_files):
        print('%d/%d' % (j, num_files))
        x=data_raw_32[j,:]
        x_len = len(x)
        x = x[: x_len - (x_len % args.scale)]

        # generate low-res version
        if args.low_pass:
            x_lr = decimate(x, args.scale)
        else:
            x_lr = np.array(x[0::args.scale])

        if args.interpolate:
            x_lr = upsample(x_lr, args.scale)
            assert len(x) % args.scale == 0
            assert len(x_lr) == len(x)
        else:
            assert len(x) % args.scale == 0
            assert len(x_lr) == len(x) / args.scale

        if args.dimension is not -1:
            # generate patches
            max_i = len(x) - d + 1
            for i in range(0, max_i, s):
                # keep only a fraction of all the patches
                u = np.random.uniform()
                if u > args.sam: continue

                if args.interpolate:
                    i_lr = i
                else:
                    i_lr = i / args.scale

                hr_patch = np.array(x[i: i + d])
                lr_patch = np.array(x_lr[i_lr: i_lr + d_lr])

                assert len(hr_patch) == d
                assert len(lr_patch) == d_lr

                hr_patches.append(hr_patch.reshape((d, 1)))
                lr_patches.append(lr_patch.reshape((d_lr, 1)))
        else:  # for full snr
            # append the entire file without patching
            x = x[:, np.newaxis]
            x_lr = x_lr[:, np.newaxis]
            hr_patches.append(x[:len(x) // 256 * 256])
            lr_patches.append(x_lr[:len(x_lr) // 256 * 256])

    if args.dimension is not -1:
        # crop # of patches so that it's a multiple of mini-batch size
        num_patches = len(hr_patches)
        num_to_keep = int(np.floor(num_patches / args.batch_size) * args.batch_size)
        hr_patches = np.array(hr_patches[:num_to_keep])
        lr_patches = np.array(lr_patches[:num_to_keep])

    if args.dimension is not -1:
        # create the hdf5 file
        data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
        label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

        data_set[...] = lr_patches
        label_set[...] = hr_patches
    else:
        # pickle the data
        pickle.dump(hr_patches, open('full-label-' + args.out[:-7], 'wb'))
        pickle.dump(lr_patches, open('full-data-' + args.out[:-7], 'wb'))


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
    # create train
    with h5py.File(args.out, 'w') as f:
        add_data(f, args.file_list, args, save_examples=False)
