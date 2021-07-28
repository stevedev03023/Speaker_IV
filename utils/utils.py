import os
import torch
import numpy as np
from scipy.io.wavfile import read
import parselmouth
from parselmouth.praat import call
import sox

def find_files(dir, pat):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(pat):
                files.append(os.path.join(dirpath, filename))
    return files


def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav

def split_into_batch(tasks, batch_size):
    batched_tasks = []
    for i in range(0, len(tasks), batch_size):
        batched_tasks.append(tasks[i:i + batch_size])
    return batched_tasks


def get_pitch(audio_fname):
    sound = parselmouth.Sound(audio_fname)
    pitch = sound.to_pitch().selected_array
    x_np = np.asarray([pitch[i][j] for i in range(pitch.shape[0]) for j in range(2)], dtype='float32').reshape(-1, 2)
    x_np[:,0]/=600
    return x_np

def makemanifest(filelist_path, mani_path):
    f = open(filelist_path, 'r')
    paths = f.readlines()
    f.close()
    wf = open(mani_path, 'w')
    for path in paths:
        path = path.strip()
        enc_file_path = path.replace(".wav", "_enc.npy")
        mel_file_path = path.replace(".wav", "_mel.npy")
        if os.path.isfile(mel_file_path) and os.path.isfile(enc_file_path):
            mel = np.load(mel_file_path)
            len = mel.shape[0]
            enc_file_path = enc_file_path.replace('\\', '/')
            mel_file_path = mel_file_path.replace('\\', '/')
            line = "{\"enc_file\":\"%s\", \"mel_file\":\"%s\", \"length\":%d}\n" %(enc_file_path, mel_file_path, len)
            wf.write(line)
    wf.close()

def makedataset_wg(filelist_path):
    f = open(filelist_path, 'r')
    paths = f.readlines()
    f.close()
    file_id=0
    for path in paths:
        pred_mel = test_decoder_model(path.strip())
        outfn = path.strip().replace(".wav", ".mel")
        mel = torch.from_numpy(np.transpose(pred_mel))
        torch.save(mel, outfn)
        file_id += 1
        print("%dth of %d , saved %s" % (file_id, len(paths), outfn))

#makemanifest("files_libri_test.txt", "manifest_libritts_test.txt")
