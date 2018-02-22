def batch2(data):
    lengths=len(data)
    data=sorted(data, key=lambda x: len(x))
    id = list(range(0, lengths, batch_size))
    id=shuffle(id)
    for i in id:
        yield data[i:(i+batch_size)]

new=list(zip(audio, train_targets))

    
tr_data=data[:40]
val_data=data[41:]