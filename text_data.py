target_text =glob(os.path.join(TMP_DIR, '*.txt'))
def huu(i):
    train_targets=[]
    with open(target_text[i], 'rt', encoding='UTF8') as f:
        line=f.readlines()
        train_targets=encode(preprocess(str(line)))    
    return train_targets

train_targets=[huu(i) for i in range(len(target_text))]

data=list(zip(train_inputs, train_targets))