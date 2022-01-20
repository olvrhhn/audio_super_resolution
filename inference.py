import os, glob

import soundfile
import torch
from scipy.signal import decimate
from scipy import interpolate
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from datasets.dataset import Sonata32Dataset, SingleSpeaker
from utils.metrics import *
from models.audiounet import AudioUNet
from models.tfilmunet import TFILMUNet
import matplotlib.pyplot as plt
import matplotlib
import torchaudio.transforms as T


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

def plot_spectrogram(spec, title=None, save_name='test.png', ylabel='frequency (Hz)', aspect='auto',y_max=16000, xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title)
  axs.set_ylabel(ylabel)
  axs.set_xlabel('timeframe (-)')
  plt.set_cmap('inferno')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  # axs.set_yscale('log')
  axs.set_ylim((0,y_max)) # was 0,1000
  # axs.set_yticks([50, 100, 200, 500, 1000, 2000, 5000,10_000,16_000])
  # axs.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  clb = plt.colorbar(im, ax=axs)
  clb.ax.set_title('dB')
  # clb.set_label(label='dB', labelpad=-40, y=1.05, rotation=0)
  # fig.colorbar(im, ax=axs, label='dB')
  im.set_clim(-40,40)

  # fig.colorbar.set_label('Db', rotation=270)

  plt.show()
  fig.savefig(save_name, bbox_inches='tight')

def plot(waveform,
         name,
         file_name,
         n_fft = 32000,
         win_length = 1024,
         hop_length = 2048,
         low_res = False):

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    # Perform transformation
    spec = spectrogram(waveform)


    if low_res:
        spec_new = torch.zeros(1,int(spec.shape[1]*3),spec.shape[2])
        spec_fill = torch.cat((spec, spec_new),1)
        print(f"spec fill min: {torch.min(spec_fill)}")
        print(f"spec fill max: {torch.max(spec_fill)}")
        plot_spectrogram(spec_fill[0], title=name, save_name=file_name, y_max=4*int(n_fft/2))

    else:
        # print_stats(spec)
        print(f"spec min: {torch.min(spec)}")
        print(f"spec max: {torch.max(spec)}")
        plot_spectrogram(spec[0], title=name, save_name=file_name, y_max=int(n_fft/2))


def main(opts):

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(f"Current training run {time_stamp} has started!")

    # Setup Meter and Device
    meter = averageMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opts.dataset_type == "vctk-single":
        test_dataset = SingleSpeaker(root=os.path.join(opts.dataset_root))
    elif opts.dataset_type == "piano":
        test_dataset = Sonata32Dataset(root='datasets/music_dataset/data/',
                                       target_type='valid',
                                       sr=16000,
                                       scale=8,
                                       dimension=8192,
                                       stride=4096)
    try:
        test_loader = DataLoader(test_dataset,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.num_workers)
    except UnboundLocalError:
        print("No dataset specified.")
        return


    # Setup model and pick checkpoint
    model = TFILMUNet()
    if os.path.exists(opts.checkpoints_root):
        checkpoint = max(glob.glob(os.path.join(opts.checkpoints_root, opts.checkpoint)), key=os.path.getctime)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    else:
        raise ValueError(f"Checkpoints directory {opts.checkpoints_root} does not exist")
    model = model.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        for test_idx, test_sample in enumerate(tqdm(test_loader)):
            # Put img and gt on GPU if available
            in_test, gt_test = test_sample[0].to(device), test_sample[2].to(device)

            # Forward pass and loss calculation
            if opts.method == "our":
                out_test = model(in_test)
            elif opts.method == "base":
                out_test = in_test

            # # Update iou meter
            meter.update(np.array(gt_test.cpu()), np.array(out_test.cpu()), opts.batch_size)


    snr, lsd = meter.get_score()
    print("\n---INFERENCE on Checkpoint: {}---\nMODE is set to {}\nSignal to Noise Ratio (SNR): {} \nLog-spectral distance (LSD): {}".format(opts.checkpoint, opts.method, round(float(snr), 4), round(float(lsd), 4)))


def run_examples(opts):
    print('Run Inference on example files:')
    # Setup model and pick checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TFILMUNet()
    if os.path.exists(opts.checkpoints_root):
        checkpoint = max(glob.glob(os.path.join(opts.checkpoints_root, opts.checkpoint)), key=os.path.getctime)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    else:
        raise ValueError(f"Checkpoints directory {opts.checkpoints_root} does not exist")
    model = model.to(device)
    # Inference
    model.eval()
    os.makedirs('examples', exist_ok=True)
    # save wav:
    with open(opts.wav_file_list) as f:
        for line in f:
            filename = os.path.basename(os.path.splitext(line)[0])
            print('Evaluating: ' + filename)
            os.makedirs('examples/' + filename, exist_ok=True)
            with torch.no_grad():
                hr, fs = librosa.load(line.strip(), sr=opts.sr)
                hr = np.pad(hr, (0, 4096 - (hr.shape[0] % 4096)), 'constant',
                              constant_values=(0, 0))
                lr = decimate(hr, opts.scale)
                lr_int = upsample(lr, opts.scale)

                # hr_t = torch.as_tensor(hr, device=device)
                lr_int_t = torch.unsqueeze(torch.as_tensor(lr_int,dtype=torch.float32, device=device),0)

                out = model(torch.unsqueeze(lr_int_t,0))

                soundfile.write('examples/' + filename + '/' + filename + '_gt.wav', hr, opts.sr)  # high res
                soundfile.write('examples/' + filename + '/' + filename + '_low_res.wav', lr, int(opts.sr / opts.scale))  # low res
                soundfile.write('examples/' + filename + '/' + filename + '_base.wav', lr_int, opts.sr)  # baseline res
                soundfile.write('examples/' + filename + '/' + filename + '_super.wav', out[0, 0, :].detach().numpy(), opts.sr)  # super res

                # plots:
                plot(waveform=torch.unsqueeze(torch.as_tensor(lr.copy(),dtype=torch.float32, device=device),0), name="Low Resolution", n_fft=int(int(opts.sr*2) / 4), hop_length=int(1024/4), win_length=int(2048/4), file_name='examples/' + filename + '/spec_low_res_' + filename + '.pdf', low_res=True)
                plot(waveform=torch.unsqueeze(torch.as_tensor(hr.copy(),dtype=torch.float32, device=device),0), name="High Resolution",n_fft=int(opts.sr*2), hop_length=1024, win_length=2048, file_name='examples/' + filename + '/spec_high_res_' + filename + '.pdf')
                plot(waveform=lr_int_t, name="Baseline",n_fft=int(opts.sr*2), hop_length=1024, win_length=2048, file_name='examples/' + filename + '/spec_baseline_' + filename + '.pdf')
                plot(waveform=torch.squeeze(out,0), name="Super Resolution", n_fft=int(opts.sr*2), hop_length=1024, win_length=2048, file_name='examples/' + filename + '/spec_super_res_' + filename + '.pdf')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="vctk-single"
    ),
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.getcwd(), "datasets", "vctk", "vctk-speaker1-val.4.16000.8192.4096.h5")
    )
    parser.add_argument(
        "--full-root",
        type=str,
        default=os.path.join(os.getcwd(), "datasets", "vctk", "test.h5")
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="val"
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="Save spectrogram plots and wav files"
    )
    parser.add_argument(
        "--wav-file-list",
        type=str,
        default="assets/save_wav_list.txt"
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=8192
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000
    ),
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default=os.path.join(os.getcwd(), "checkpoints", "runs")
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="2021_06_23_17_39_44.pth"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=["base", "our"],
        help="switch between BASELINE and OUR Method"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inf",
        choices=["inf", "vis"],
        help="switch between inference and visualization mode"
    )
    clargs = parser.parse_args()
    print(clargs)

    if clargs.save_examples:
        run_examples(clargs)
    else:
        main(clargs)
