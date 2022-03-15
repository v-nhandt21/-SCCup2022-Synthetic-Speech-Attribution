import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from specAugment import spec_augment_pytorch
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
     sampling_rate, data = read(full_path)
     return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
     return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
     return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
     output = dynamic_range_compression_torch(magnitudes)
     return output


def spectral_de_normalize_torch(magnitudes):
     output = dynamic_range_decompression_torch(magnitudes)
     return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
     if torch.min(y) < -1.:
          print('min value is ', torch.min(y))
     if torch.max(y) > 1.:
          print('max value is ', torch.max(y))

     global mel_basis, hann_window
     if fmax not in mel_basis:
          mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
          mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
          hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
     y = y.squeeze(1)

     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                         center=center, pad_mode='reflect', normalized=False, onesided=True)

     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

     spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
     spec = spectral_normalize_torch(spec)

     return spec


def get_dataset_filelist(a):
     with open(a.input_training_file, 'r', encoding='utf-8') as fi:
          training_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]

     with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
          validation_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]

     return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
     def __init__(self, training_files, segment_size, n_fft, num_mels,
                    hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                    device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, evaluation = False):
          self.audio_files = training_files
          random.seed(1234)
          if shuffle:
               random.shuffle(self.audio_files)
          self.segment_size = segment_size
          self.sampling_rate = sampling_rate
          self.split = split
          self.n_fft = n_fft
          self.num_mels = num_mels
          self.hop_size = hop_size
          self.win_size = win_size
          self.fmin = fmin
          self.fmax = fmax
          self.fmax_loss = fmax_loss
          self.cached_wav = None
          self.n_cache_reuse = n_cache_reuse
          self._cache_ref_count = 0
          self.device = device
          self.fine_tuning = fine_tuning
          self.base_mels_path = base_mels_path
          self.evaluation = evaluation

     def __getitem__(self, index):
          filename, label = self.audio_files[index]
          if self._cache_ref_count == 0:
               audio, sampling_rate = load_wav(filename)
               audio = audio / MAX_WAV_VALUE
               if not self.fine_tuning:
                    audio = normalize(audio) * 0.95
               self.cached_wav = audio
               if sampling_rate != self.sampling_rate:
                    raise ValueError("{} SR doesn't match target {} SR".format(
                         sampling_rate, self.sampling_rate))
               self._cache_ref_count = self.n_cache_reuse
          else:
               audio = self.cached_wav
               self._cache_ref_count -= 1

          audio = torch.FloatTensor(audio)
          audio = audio.unsqueeze(0)

          if audio.size(1) >= self.segment_size:
               max_audio_start = audio.size(1) - self.segment_size
               audio_start = random.randint(0, max_audio_start)
               audio = audio[:, audio_start:audio_start+self.segment_size]
          else:
               audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

          mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                              center=False)

          mel = spec_augment_pytorch.spec_augment(mel_spectrogram=mel,
               time_warping_para=80, 
               frequency_masking_para=27,
               time_masking_para=100, 
               frequency_mask_num=1, 
               time_mask_num=1)

          if self.evaluation:
               np.save(filename.replace("wav","npy"), mel.repeat(3,1,1).permute(1,2,0).detach().numpy())

          return (mel.squeeze(), audio.squeeze(0), filename, int(label))

     def __len__(self):
          return len(self.audio_files)

if __name__ == '__main__':
     from torch.utils.data import DistributedSampler, DataLoader
     import argparse, json, shutil
     device = "cuda"
     class AttrDict(dict):
          def __init__(self, *args, **kwargs):
               super(AttrDict, self).__init__(*args, **kwargs)
               self.__dict__ = self


     def build_env(config, config_name, path):
          t_path = os.path.join(path, config_name)
          if config != t_path:
               os.makedirs(path, exist_ok=True)
               shutil.copyfile(config, os.path.join(path, config_name))

     parser = argparse.ArgumentParser()

     parser.add_argument('--config', default='config.json')
     parser.add_argument('--input_testing_file', default='Static/data/test.txt')
     parser.add_argument('--input_mels_dir', default='')
     parser.add_argument('--fine_tuning', default=False, type=bool)


     a = parser.parse_args()

     with open(a.config) as f:
          data = f.read()

     json_config = json.loads(data)
     h = AttrDict(json_config)

     with open(a.input_testing_file, 'r', encoding='utf-8') as fi:
          testing_files = [x.split(',') for x in fi.read().split('\n') if len(x) > 0]

     validset = MelDataset(testing_files, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir, evaluation = True)
     validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)
     
     for j, batch in enumerate(validation_loader):
          mel, audio, _, label = batch