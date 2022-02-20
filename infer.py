import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import shutil
import sys

class AttrDict(dict):
         def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self


def build_env(config, config_name, path):
     t_path = os.path.join(path, config_name)
     if config != t_path:
          os.makedirs(path, exist_ok=True)
          shutil.copyfile(config, os.path.join(path, config_name))

h = None
device = None
from efficientnet_pytorch import EfficientNet

def load_checkpoint(filepath, device):
     assert os.path.isfile(filepath)
     print("Loading '{}'".format(filepath))
     checkpoint_dict = torch.load(filepath, map_location=device)
     print("Complete.")
     return checkpoint_dict


def get_mel(x):
     return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
     pattern = os.path.join(cp_dir, prefix + '*')
     cp_list = glob.glob(pattern)
     if len(cp_list) == 0:
          return ''
     return sorted(cp_list)[-1]


def inference(model_path = "Static/checkpoint/g_00015000"):
     generator = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(device)

     state_dict_g = torch.load(model_path, map_location=device)

     generator.load_state_dict(state_dict_g['generator'])

     generator.eval()

     Count = 0

     with torch.no_grad():
          with open("Static/data/val.txt", "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in lines:
                    file, label = line.split(",")

                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    predict = generator(x.unsqueeze(1).repeat(1,3,1,1))
                    predict = predict.squeeze()

                    if int(torch.argmax(predict).detach().cpu().numpy()) == int(label):
                         Count += 1
                    else:
                         print(line)
               print(Count/len(lines))

def main(model_path):

     config_file = 'config.json'
     with open(config_file) as f:
          data = f.read()

     global h
     json_config = json.loads(data)
     h = AttrDict(json_config)

     torch.manual_seed(h.seed)
     global device
     if torch.cuda.is_available():
          torch.cuda.manual_seed(h.seed)
          device = torch.device('cuda')
     else:
          device = torch.device('cpu')

     inference(model_path)


if __name__ == '__main__':

     if len(sys.argv) > 1:
          model_path = sys.argv[1]
     else:
          model_path = "Static/checkpoint/g_00015000"

     main(model_path)