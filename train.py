import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from utils import save_checkpoint
import os
import shutil

from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

def save_checkpoint(filepath, obj):
     print("Saving checkpoint to {}".format(filepath))
     torch.save(obj, filepath)
     print("Complete.")

def build_env(config, config_name, path):
     t_path = os.path.join(path, config_name)
     if config != t_path:
          os.makedirs(path, exist_ok=True)
          shutil.copyfile(config, os.path.join(path, config_name))

torch.backends.cudnn.benchmark = True


def train(rank, a, h):

     torch.cuda.manual_seed(h.seed)
     device = torch.device('cuda:{:d}'.format(rank))

     model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(device)

     if rank == 0:
          os.makedirs(a.checkpoint_path, exist_ok=True)

     steps = 0
     state_dict_do = None
     last_epoch = -1

     optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate)

     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

     training_filelist, validation_filelist = get_dataset_filelist(a)

     trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                              fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

     train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

     train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                                   sampler=train_sampler,
                                   batch_size=h.batch_size,
                                   pin_memory=True,
                                   drop_last=True)

     if rank == 0:
          validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                                   h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                                   fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                                   base_mels_path=a.input_mels_dir)
          validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                        sampler=None,
                                        batch_size=1,
                                        pin_memory=True,
                                        drop_last=True)

          sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

     
     model.train()
     criterion = torch.nn.CrossEntropyLoss()

     for epoch in tqdm(range(max(0, last_epoch), a.training_epochs)):
          # if rank == 0:
          #      start = time.time()
          #      print("Epoch: {}".format(epoch+1))

          for i, batch in enumerate(train_loader):
               if rank == 0:
                    start_b = time.time()
               mel, audio, _, label = batch
               x = torch.autograd.Variable(mel.to(device, non_blocking=True))
               y = torch.autograd.Variable(audio.to(device, non_blocking=True))

               label = torch.Tensor.long(label).to(device)

               y = y.unsqueeze(1)

               y_g_hat = model(x.unsqueeze(1).repeat(1,3,1,1))

               loss = criterion(y_g_hat, label)

               optim_g.zero_grad()

               loss.backward()
               optim_g.step()

               if rank == 0:
                    # STDOUT logging
                    # if steps % a.stdout_interval == 0:

                    #      print('Steps : {:d}, Gen Loss Total : {:4.3f}, s/b : {:4.3f}'.
                    #           format(steps, loss, time.time() - start_b))

                    # checkpointing
                    if steps % a.checkpoint_interval == 0 and steps != 0:
                         checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                         save_checkpoint(checkpoint_path,
                                        {'generator': (model.module if h.num_gpus > 1 else model).state_dict()})

                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                         sw.add_scalar("training/gen_loss_total", loss, steps)

                    # Validation
                    if steps % a.validation_interval == 0:  # and steps != 0:
                         model.eval()
                         torch.cuda.empty_cache()
                         val_err_tot = 0
                         with torch.no_grad():
                              for j, batch in enumerate(validation_loader):
                                   mel, audio, _, label = batch
                                   x = torch.autograd.Variable(mel.to(device, non_blocking=True))
                                   y = torch.autograd.Variable(audio.to(device, non_blocking=True))

                                   label = torch.Tensor.long(label).to(device)

                                   y_g_hat = model(x.unsqueeze(1).repeat(1,3,1,1))

                                   loss = criterion(y_g_hat, label)

                         sw.add_scalar("validation/loss", loss, steps)

                         model.train()

               steps += 1

          scheduler_g.step()
          
          # if rank == 0:
          #      print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
     print('Initializing Training Process..')

     parser = argparse.ArgumentParser()

     parser.add_argument('--group_name', default=None)
     parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
     parser.add_argument('--input_mels_dir', default='')
     parser.add_argument('--input_training_file', default='Static/data/train.txt')
     parser.add_argument('--input_validation_file', default='Static/data/val.txt')
     parser.add_argument('--checkpoint_path', default='Static/checkpoint')
     parser.add_argument('--config', default='config.json')
     parser.add_argument('--training_epochs', default=250, type=int)
     parser.add_argument('--stdout_interval', default=5, type=int)
     parser.add_argument('--checkpoint_interval', default=5000, type=int)
     parser.add_argument('--summary_interval', default=100, type=int)
     parser.add_argument('--validation_interval', default=1000, type=int)
     parser.add_argument('--fine_tuning', default=False, type=bool)

     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     json_config = json.loads(data)
     h = AttrDict(json_config)
     build_env(a.config, 'config.json', a.checkpoint_path)

     torch.manual_seed(h.seed)
     if torch.cuda.is_available():
          torch.cuda.manual_seed(h.seed)
          h.num_gpus = torch.cuda.device_count()
          h.batch_size = int(h.batch_size / h.num_gpus)
          print('Batch size per GPU :', h.batch_size)
     else:
          pass

     if h.num_gpus > 1:
          mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
     else:
          train(0, a, h)


if __name__ == '__main__':
     main()
     # loss = torch.nn.CrossEntropyLoss()
     # input = torch.randn(3, 5, requires_grad=True)
     # target = torch.empty(3, dtype=torch.long).random_(5)

     # print(input.size())
     # print(target.size())
     # print(target)
     # output = loss(input, target)
     # print(output)