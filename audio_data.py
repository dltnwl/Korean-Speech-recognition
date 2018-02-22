import pickle
import os.path
import os
import soundfile as sf
from abc import ABC, abstractmethod
from scipy.signal import resample_poly
from python_speech_features import mfcc
import numpy as n
from glob import glob
import time
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from six.moves import xrange as range
import random

_DECODE_MAP = LABELS + [""]
_ENCODE_MAP = {c: i for i, c in enumerate(_DECODE_MAP)}
def encode(string):
    return list(map(lambda c: _ENCODE_MAP[c], string))
def decode(sequence):
    return "".join(map(lambda c: _DECODE_MAP[c], sequence))

TMP_DIR='C:\\Users\\user\\Downloads\\small_krs\\test\\fv03'

files=glob(os.path.join(TMP_DIR, '*.wav'))

def audio_only(f):
    return [sf.read(audio_path)for audio_path in f]
audio=audio_only(files)
def mfcc_list(audio):
    return [np.asarray(mfcc(audio[i][0], audio[i][1])[np.newaxis, :]) for i in range(0,len(audio))]
train_inputs=mfcc_list(audio)