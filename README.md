Korean speech recognition(End to End)
=================


CTC
---------------------------
  * Noise 여부에 따른 BiLSTM / CNN+BiLSTM 트레이닝 추이
 
Model| Test LER(문장) | 
---- | ---- | 
CNN-LSTM(256) | 0.277151 | 
CNN-BiLSTM(256) | 0.225424 |
BiLSTM(256) with noise | 0.4920 |
CNN-BiLSTM(256) with noise | 0.3956 |


<img src="/image/picture.JPG" width="400" height="200">

[http://proceedings.mlr.press/v32/graves14.pdf](http://proceedings.mlr.press/v32/graves14.pdf)


Listen Attend and Spell
---------------------------

[https://arxiv.org/abs/1508.01211](https://arxiv.org/abs/1508.01211)


Result(단어)
---------------------------
 
Model| LER | WER |
---- | ---- | ---- |
CTC| 5.092 | 33.00 |
Attention | 5.750 | 30.40 |


 *실제와 추정 값 비교
 
Original| Decode | 
---- | ---- | 
저축되다 | 저축되다 | 
늑혼하다 | 늑콩하다 |
고물상 | 고물상 |
한몫 | 한목 |
수출입 | 수추립 |
용량 | 용량 |
개울물 | 개울물 |



