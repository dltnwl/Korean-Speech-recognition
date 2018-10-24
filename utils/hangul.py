import re



INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
"char list: Hangul initials (초성)"



MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
"char list: Hangul medials (중성)"



FINALS = list("∅ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
"char list: Hangul finals (종성)."



SPACE_TOKEN = " "
LABELS = sorted({SPACE_TOKEN}.union(INITIALS).union(MEDIALS).union(FINALS))

"char list: All CTC labels."





def check_syllable(char):
  return 0xAC00 <= ord(char) <= 0xD7A3



def split_syllable(char):
  assert check_syllable(char)
  diff = ord(char) - 0xAC00
  _m = diff % 28
  _d = (diff - _m) // 28
  return (INITIALS[_d // 21], MEDIALS[_d % 21], FINALS[_m])



def preprocess(str):
  result = ""
  for char in re.sub("\\s+", SPACE_TOKEN, str.strip()):
    if char == SPACE_TOKEN:
      result += SPACE_TOKEN
    elif check_syllable(char):
      result += "".join(split_syllable(char))
  return result
