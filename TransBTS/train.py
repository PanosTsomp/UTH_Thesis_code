import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch import nn

from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models import criterions
from data.BraTS import BraTS
from tensorboardX import SummaryWriter

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# ----------------------------------------------------------------------------------------
# Basic Information
# ----------------------------------------------------------------------------------------
parser.add_argument('--user', default='name of user', type=str)
parser.add_argument('--experiment', default='TransBTS', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description', 
                    default='TransBTS, training on train.txt!', 
                    type=str)

# ----------------------------------------------------------------------------------------
# DataSet Information
# ----------------------------------------------------------------------------------------
parser.add_argument('--root', default='path to training set', type=str)
parser.add_argument('--train_dir', default='Train', type=str)
parser.add_argument('--valid_dir', default='Valid', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--valid_file', default='valid.txt', type=str)
parser.add_argument('--dataset', default='brats', type=str)
parser.add_argument('--model_name', default='TransBTS', type=str)
parser.add_argument('--input_C', default=4, type=int)
parser.add_argument('--input_H', default=240, type=int)
parser.add_argument('--input_W', default=240, type=int)
parser.add_argument('--input_D', default=160, type=int)
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--output_D', default=155, type=int)

# ----------------------------------------------------------------------------------------
# Training Information
# ----------------------------------------------------------------------------------------
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='softmax_dice', type=str)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str, 
                    help="Which GPU to use (e.g. '0'); single GPU only.")
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=1000, type=int)
parser.add_argument('--save_freq', default=1000, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)

args = parser.parse_args()

def main():
    # ------------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------------
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                           'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('-------------------------------------- All Configs --------------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('------------------------------------------------------------------------------------------')
    logging.info('{}'.format(args.description))

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If you only have one GPU, choose device = 'cuda:0' if available
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Requires at least one CUDA device"
    device = torch.device("cuda:0")  # single GPU

    # ------------------------------------------------------------------------
    # Create Model
    # ------------------------------------------------------------------------
    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    model = model.to(device)
    model.train()

    # ------------------------------------------------------------------------
    # Create Optimizer and Criterion
    # ------------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        amsgrad=args.amsgrad
    )
    criterion = getattr(criterions, args.criterion)

    # ------------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------------
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                  'checkpoint', args.experiment + args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = args.resume
    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('Successfully loaded checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('No checkpoint found or load=False -> re-training from scratch!')

    writer = SummaryWriter()

    # ------------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------------
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, "")

    logging.info(f"Training list file: {train_list}")
    logging.info(f"Training root path: {train_root}")

    train_set = BraTS(train_list, train_root, args.mode)
    logging.info('Samples for train = {}'.format(len(train_set)))

    # Standard DataLoader with shuffle=True since single GPU
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    start_time = time.time()

    # ------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.end_epoch):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch_time = time.time()

        # Adjust the learning rate for this epoch
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

        for i, data in enumerate(train_loader):
            x, target = data
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward
            output = model(x)

            # Compute loss
            loss, loss1, loss2, loss3 = criterion(output, target)
            # Directly use loss.item() for logging on single GPU
            loss_val   = loss.item()
            loss1_val  = loss1.item()
            loss2_val  = loss2.item()
            loss3_val  = loss3.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if i % 20 == 0:  # Example: log every 20 steps
                logging.info('Epoch: {} | Iter: {} | loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f}'
                             .format(epoch, i, loss_val, loss1_val, loss2_val, loss3_val))

        end_epoch_time = time.time()

        # Save checkpoint if needed
        if (epoch + 1) % int(args.save_freq) == 0 \
           or (epoch + 1) in [args.end_epoch - 1, args.end_epoch - 2, args.end_epoch - 3]:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, file_name)

        # Write to TensorBoard
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('loss', loss_val, epoch)
        writer.add_scalar('loss1', loss1_val, epoch)
        writer.add_scalar('loss2', loss2_val, epoch)
        writer.add_scalar('loss3', loss3_val, epoch)

        # Print time consumption
        epoch_time_minute = (end_epoch_time - start_epoch_time) / 60.0
        remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60.0
        logging.info('Epoch {} finished in {:.2f} minutes.'.format(epoch, epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours.'.format(remaining_time_hour))

    # ------------------------------------------------------------------------
    # Final Model Save
    # ------------------------------------------------------------------------
    writer.close()
    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    }, final_name)

    total_time = (time.time() - start_time) / 3600.0
    logging.info('Total training time is {:.2f} hours.'.format(total_time))
    logging.info('------------------ Training process finished! ------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    """
    Polynomial decay of learning rate, typical for some segmentation tasks.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - float(epoch) / float(max_epoch), power), 8)


def log_args(log_file):
    """
    Basic logger setup: logs both to file and console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s ===> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    # Use main() for single-GPU (no distributed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
