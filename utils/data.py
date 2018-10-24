
import re as _re
import os as _os
import json as _json
import os.path as _path
import soundfile as _sf
import glob

def _rawdata_loader(f):
  def decorated(data_dir, **kwargs):
    n = 0
    for (audio_path, text) in f(data_dir, **kwargs):
      try:
        yield (_sf.read(audio_path), text)
        n += 1
        if n % 500 == 0:
          print("%d files loaded." % n)
      except Exception as e:
        print(e)
  return decorated

@_rawdata_loader
def opendict(data_dir, **kwargs):
  """Load dictionary pronounciation data."""
  join = lambda f: _path.join(data_dir, f)
  with open(join("list.jl")) as f:
    for e in filter(lambda x: x.get("files"), map(_json.loads, f)):
      yield (join(e["files"][0]["path"]), e["word"].strip())


@_rawdata_loader
def zeroth(data_dir):

    def download_data(part_dir):
        join = lambda f: _path.join(part_dir, f)
        #if _re.match(r'[0-9]',_path.basename(part_dir)) is None:

        
        with open(join("{0}_003.trans.txt".format(_path.basename(part_dir))), encoding="utf-8") as f:
            path = []
            tmp=[]
            for line in f.readlines():
                p = _re.compile(r'[0-9]+[_][0-9]+[_][0-9]+')
                num=p.match(line)
                id=num.group()
                
                p = _re.compile('[^0-9]+[^_0-9].')
                lab=p.findall(line)
                tmp.append(lab[0][1:])
            
                #flac파일 
                audio_path= join("{0}.flac".format(id))
                path.append(audio_path)
            return path, tmp  
        
    audio_path=[]
    text=[]
    for i in list(glob.glob(_path.join(data_dir,'*'))):
        if _re.match(r'[0-9]+',_path.basename(i)):
            label, tmp=download_data(i)
            audio_path.extend(label)
            text.extend(tmp)
 
    for tup in list(zip(audio_path, text)):
        yield (tup[0], tup[1])
