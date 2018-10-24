import numpy as _np
from scipy.signal import resample_poly as _resample_poly
from python_speech_features import mfcc as _mfcc

SAMPLE_RATE = 16000

def mfcc(audio, samplerate, n_features, **kwargs):
  mfcc = _mfcc(audio, samplerate=samplerate, numcep=n_features, **kwargs)
  return mfcc
