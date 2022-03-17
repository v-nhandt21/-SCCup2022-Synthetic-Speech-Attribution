import glob
import os
import argparse
import json
from re import T
import torch
from scipy.io.wavfile import write
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
import shutil
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
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
               for line in tqdm(lines):
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

def prediction(model_path = "Static/checkpoint/g_00050000", wav_dir = "Static/evaluation/part2/", predict_file="Static/evaluation/labels_part2.txt", threshold_unseen=False):

     if threshold_unseen:
          generator = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(device)
     else:
          generator = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6).to(device)
     state_dict_g = torch.load(model_path, map_location=device)
     generator.load_state_dict(state_dict_g['generator'])
     generator.eval()

     if threshold_unseen:
          Threshold = []
          with torch.no_grad():
               unseen_file = open("Static/data/unseen.txt", "r", encoding="utf-8")
               lines = unseen_file.read().splitlines()
               for line in tqdm(lines[:500]):
                    file = "Static/data/unseen/" + line.split(",")[0]
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    predict = generator(x.unsqueeze(1).repeat(1,3,1,1))
                    predict = predict.squeeze()
                    Threshold.append(float(max(predict)))
          print("Max threshold for unseen data: ", max(Threshold))
          plt.plot(Threshold)
          Threshold = sum(Threshold)/len(Threshold)
          print("Use average threshold for unseen data: ", Threshold)
          

          Threshold_seen = []
          with torch.no_grad():
               unseen_file = open("Static/data/val.txt", "r", encoding="utf-8")
               lines = unseen_file.read().splitlines()
               for line in tqdm(lines):
                    file = line.split(",")[0]
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    predict = generator(x.unsqueeze(1).repeat(1,3,1,1))
                    predict = predict.squeeze()
                    Threshold_seen.append(float(max(predict)))
          print("Min threshold for seen data: ", min(Threshold_seen))
          plt.plot(Threshold_seen)
          Threshold_seen = sum(Threshold_seen)/len(Threshold_seen)
          print("Average threshold for seen data: ", Threshold_seen)

          plt.savefig("threshold.png")

     with torch.no_grad():
          fw = open("Static/evaluation/answer.txt", "w", encoding="utf-8")
          with open(predict_file, "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in tqdm(lines):
                    file = wav_dir + line
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    predict = generator(x.unsqueeze(1).repeat(1,3,1,1))
                    predict = predict.squeeze()
                    # print(predict)

                    if threshold_unseen:
                         if max(predict)>12:
                              fw.write( line+", "+str(int(torch.argmax(predict).detach().cpu().numpy()))+"\n" )
                         else:
                              fw.write( line+", 5\n" )
                    else:
                         fw.write( line+", "+str(int(torch.argmax(predict).detach().cpu().numpy()))+"\n" )

def cascade(model_path_unseen = "Static/checkpoint_unseen/g_00050000", model_path_algo = "Static/checkpoint_algorithm/g_00050000", wav_dir = "Static/evaluation/part1/", predict_file="Static/evaluation/labels_part1.txt"):
     
     generator_unseen = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2).to(device)
     state_dict_g = torch.load(model_path_unseen, map_location=device)
     generator_unseen.load_state_dict(state_dict_g['generator'])
     generator_unseen.eval()

     generator_algo = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(device)
     state_dict_g = torch.load(model_path_algo, map_location=device)
     generator_algo.load_state_dict(state_dict_g['generator'])
     generator_algo.eval()

     with torch.no_grad():
          fw = open("Static/evaluation/answer.txt", "w", encoding="utf-8")
          with open(predict_file, "r", encoding="utf-8") as f:
               lines = f.read().splitlines()
               for line in tqdm(lines):
                    file = wav_dir + line
                    wav, sr = load_wav(file)
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    
                    check = generator_unseen(x.unsqueeze(1).repeat(1,3,1,1))
                    check = check.squeeze()
                    if int(torch.argmax(check).detach().cpu().numpy()) == 1:
                         fw.write( line+", 5\n" )
                         continue

                    predict = generator_algo(x.unsqueeze(1).repeat(1,3,1,1))
                    predict = predict.squeeze()
                    fw.write( line+", "+str(int(torch.argmax(predict).detach().cpu().numpy()))+"\n" )
def main():

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

     cascade()


if __name__ == '__main__':
     main()