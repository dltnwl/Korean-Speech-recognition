import editdistance as ed
from glob import glob
import re as re
import os as os
import json as _json
import os.path as path
import soundfile as sf
import utils as utils
import random as _random
import utils.hangul as _hangul

from random import shuffle as shuffle
from importlib import import_module as import_module
from functools import wraps
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from six.moves import xrange as range

from abc import ABC, abstractmethod

import pandas as pd
import random as random
import tensorflow as tf
from tqdm import tqdm


import pickle as _pickle
import os.path as _path
from random import shuffle as _shuffle
import numpy as _np

learning_rate=0.0001
n_classes=56

batch_size=4
feat_len =13
max_label_len=26   #max_label_len+2해서 sos eos 추가
max_time_step=448
#pretrained=True

ctc_labels = _hangul.JAMOS
labels = [" "] + ctc_labels



###################test####################################

dfa = {0:[-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0],
  1:[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  2:[-1,-1,-1, 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2],
  3:[ -1,-1, 0,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 3]}


def accepts(transitions,initial,accepting,s):
    state = initial
    for c in s:
        try:
            state = transitions[state][c]
            if state==-1:
                break
        except:
            return False
    return state in accepting





class WER:
    def __init__(self, true_data, pred_data):
        self.data1 = true_data
        self.data2 = pred_data

    def sep(self,text):   
        mo=[]
        new=[]
        for i in text:
            if i!=2:
                new.append(i)
            else:
                mo.append(new)
                new=[]
        mo.append(new)
        return mo
    
    def count2(self, text):
        count=0
        for i in text:
            if i==2:
                count+=1        
        return count
    
    def editDistance(self,r, h):
    
        d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.float32).reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0: 
                    d[0][j] = j
                elif j == 0: 
                    d[i][0] = i
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitute = d[i-1][j-1] + 1
                    insert = d[i][j-1] + 1
                    delete = d[i-1][j] + 1
                    d[i][j] = min(substitute, insert, delete)
        return d[len(r)][len(h)]
    
    def wer(self):

            
        score= self.editDistance(self.sep(self.data1),self.sep(self.data2))
          
        total_wer=score/len(self.sep(self.data1))
        return total_wer




from math import log
from numpy import array
from numpy import argmax
 
# beam search
def beam_search_decoder(data, K):
   sequences = [[list(), 1.0]]
   # walk over each step in sequence
   for row in data:
      all_candidates = list()
      # expand each current candidate
      for i in range(len(sequences)):
         seq, score = sequences[i]
         for j in range(len(row)):
            candidate = [seq + [j], -score *log(row[j])]
            all_candidates.append(candidate)
      # order all candidates by score
      ordered = sorted(all_candidates, key=lambda tup:tup[1])
      sequences = ordered[:K]
   return sequences



test_ler=0.0
test_wer=0.0
label=[]
count=0
for batch in batchs["test"]:
    audio_test, label_test=preprocess(batch)
    te_loss,te_ler ,te_pred_y, te_true_y,pred_matrix=model(audio_test, label_test, False)
    for i in range(len(batch)):
        if accepts(dfa,0,{3},compress(te_pred_y[i]))==True:
             new=compress(te_pred_y[i])
             wer=WER(compress(te_true_y[i]),new)
             test_wer+=wer.wer()
             test_ler+=te_ler
             ori=_hangul.postprocess(decode(compress(te_true_y[i])))
             decoded=_hangul.postprocess(decode(new))
             count+=1
             print('original', ori)
             print('decode', decoded)

        else:
             data = pred_matrix.cpu().data.numpy()[i*122:(i+1)*122,:]
             voca_list=sep(compress(te_pred_y[i]))
             voca_num=list(map(lambda x: len(x), voca_list))
             k=1
             sentence=[]
             for idx,num in enumerate(voca_num):
     
                 if num==0:
                     pass
                 elif accepts(dfa,0,{3},voca_list[idx])==True:
                     sentence.extend([2]+voca_list[idx])
                     k+=(num+1)
                 else:
                     beam= beam_search_decoder(data[k:k+num,:], 20)#맨앞 sos랑 ' '고려                      
                     for j in range(20):
                         if accepts(dfa,0,{3},beam[j][0])==True:
                             sentence.extend([2]+beam[j][0])
                             break
                             k+=(num+1) 
                 wer=WER(compress(te_true_y[i]),sentence[1:])
                 test_wer+=wer.wer()
                 test_ler+=te_ler
                 ori=_hangul.postprocess(decode(compress(te_true_y[i])))
                 try:
                     decoded=_hangul.postprocess(decode(sentence))
                 except:
                     pass
                 count+=1
             print('original', ori[1:])
             print('decode', decoded)   

                
tt_ler=test_ler/count                        
tt_wer=test_wer/count
print('wer',tt_wer)
print('ler',tt_ler)


    for j in range(0,len(te_pred_y)):
        if accepts(dfa,0,{3},compress(te_pred_y[j]))!=True:
       #arr=np.where(arr.data.numpy()>=torch.max(arr).data.numpy(), -inf, arr.data.numpy())
    
            final=te_pred_y[j]
            aa=[]
            while accepts(dfa,0,{3},compress(final))!=True:
                id=accepts(dfa,0,{3},compress(final))
                if id!=False:
                    aa=pred_y[j*26:(j+1)*26][id].data.numpy()
                    aa=np.where(aa>=max(aa),-inf,aa)
                
                    pred_y.data.numpy()[(j*26)+id,:]=aa
                    final=torch.max(pred_y[j*26: (j+1)*26],dim=1)[1].cpu().data.numpy().reshape(26)
                else:
                    break
            aa=WER(te_true_y[j],final)
            wer+=aa.wer()
    else:
        final=te_pred_y[j]
        aa=WER(te_true_y[j],final)
        wer+=aa.wer()
    test_wer.append(wer)
    test_pred.append(final)









