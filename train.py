import os, pathlib, glob
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

from loss.super_res import MSE
from datasets.dataset import SingleSpeaker, Sonata32Dataset
from utils.metrics import *
from models.audiounet import AudioUNet
from models.tfilmunet import TFILMUNet



def main(opts):

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(f"Current training run {time_stamp} has started!")

    # Add tensorboard writer and checkpoints results directory
    writer = SummaryWriter(log_dir=os.path.join(opts.runs_root, time_stamp))
    pathlib.Path(opts.checkpoints_root).mkdir(parents=True, exist_ok=True)
    checkpoints_path = os.path.join(opts.checkpoints_root,
                                    time_stamp + ".pth")

    # Setup Metricsa and Device
    meter = averageMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup dataset
    if opts.dataset_type == "vctk-single":
        train_dataset = SingleSpeaker(root=opts.dataset_root)
        val_dataset = SingleSpeaker(root=opts.dataset_root.replace("-train.", "-val."))
    elif opts.dataset_type == "piano":
        train_dataset = Sonata32Dataset(root='datasets/music_dataset/data/',
                                       target_type='train',
                                       sr=16000,
                                       scale=4,
                                       dimension=8192,
                                       stride=4096)
        val_dataset = Sonata32Dataset(root='datasets/music_dataset/data/',
                                       target_type='valid',
                                       sr=16000,
                                       scale=4,
                                       dimension=8192,
                                       stride=4096)
    try:
        train_loader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True,
                                  num_workers=opts.num_workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=opts.num_workers)
    except UnboundLocalError:
        print("No dataset specified.")
        return

    # Setup model
    model = TFILMUNet()
    if opts.resume:
        if os.path.exists(opts.checkpoints_root):
            checkpoint = max(glob.glob(os.path.join(opts.checkpoints_root, opts.checkpoint)), key=os.path.getctime)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        else:
            raise ValueError(f"Checkpoints directory {opts.checkpoints_root} does not exist")
    model = model.to(device)

    # Setup lr scheduler, optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=opts.lr)
    criterion = MSE().to(device)


    # Training
    start = 0
    if opts.resume: start = int(os.path.split(checkpoint)[-1][-6:-4])
    for epoch in tqdm(range(start, opts.num_epochs)):
        print("\nEpoch: ", epoch)
        loss_train_epoch, loss_val_epoch = 0, 0
        for train_idx, train_sample in enumerate(train_loader):

            # Put img and gt on GPU if available
            inpt_train, gt_train = train_sample[0].to(device), train_sample[2].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass and optimization
            out_train = model(inpt_train)
            loss_train = criterion(out_train, gt_train)
            loss_train_epoch += loss_train
            loss_train.backward()
            optimizer.step()

            # Determine current lr
            writer_idx = train_idx + len(train_loader)*epoch

            # Add current loss to tensorboard
            writer.add_scalar("training_loss_step",
                              loss_train,
                              writer_idx)
            writer.add_scalar("learning_rate",
                              optimizer.param_groups[0]['lr'],
                              writer_idx)

        # Add mean epoch loss to tensorboard
        writer.add_scalar("training_loss_epoch",
                          loss_train_epoch/len(train_loader),
                          epoch)

        # Validation
        if epoch >= opts.validation_start and epoch % opts.validation_step == 0:
            model.eval()
            with torch.no_grad():
                for val_idx, val_sample in enumerate(val_loader):
                    # Put img and gt on GPU if available
                    in_val, gt_val = val_sample[0].to(device), val_sample[2].to(device)
                    # Forward pass and loss calculation
                    out_val = model(in_val)
                    loss_val = criterion(out_val, gt_val)
                    loss_val_epoch += loss_val
                    # Update iou meter
                    meter.update(np.array(gt_val.cpu()), np.array(out_val.cpu()), opts.batch_size)
            model.train()

            # Update metrics, add to tensorboard and reset
            snr, lsd = meter.get_score()
            print("\nSignal to Noise Ratio (SNR): {} \nLog-spectral distance (LSD): {}".format(round(float(snr), 4), round(float(lsd), 4)))

            writer.add_scalar("snr", snr, epoch)
            writer.add_scalar("lsd", lsd, epoch)
            meter.reset()
            # Add loss to tensorboard
            writer.add_scalar("validation_loss", loss_val_epoch/len(val_loader), epoch)

            # Save model
            if epoch == opts.validation_start or (opts.resume == True and epoch == start):
                mean_snr_best = snr
                mean_lsd_best = lsd
            mean_snr_epoch = snr
            mean_lsd_epoch = lsd
            if mean_snr_epoch >= mean_snr_best and mean_lsd_epoch <= mean_lsd_best:
                if os.path.exists(checkpoints_path):
                    os.remove(checkpoints_path)
                print("Saving Checkpoint ...")
                torch.save(model.state_dict(), checkpoints_path)
                mean_snr_best = mean_snr_epoch
                mean_lsd_best = mean_lsd_epoch
            if epoch != 0 and epoch%50 == 0:
                print("Saving Checkpoint ...")
                torch.save(model.state_dict(), checkpoints_path.replace(".pth","_epoch_"+str(epoch)+".pth"))


if __name__ == '__main__':
    parser = ArgumentParser()
    # Then input here the dataset-root as the path to ../dataset/split
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="vctk-single"
    ),
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.getcwd(), "datasets", "vctk", "vctk-speaker1-train.4.16000.8192.4096.h5")
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default=os.path.join(os.getcwd(), "checkpoints", "runs")
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=os.path.join(os.getcwd(), "runs")
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--validation-start",
        type=int,
        default=0
    )
    parser.add_argument(
        "--validation-step",
        type=int,
        default=1
    )
    # Batch size at least 2 for torchvision DeepLab Batchnorm layers
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6
    )
    # Optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=3*10e-4
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="2021_06_17_12_54_13_epoch_50.pth"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Train on from Checkpoint -> Need to provide Checkpoint Name"
    )
    clargs = parser.parse_args()
    print(clargs)
    main(clargs)