def step(data):
    audio_data=list(map(lambda x:x[0], data))
    text=list(map(lambda x:x[1], data))
    
    train_targets=sparse_tuple_from(text)
    
    
    audio_new=[np.asarray(mfcc(audio_data[i][0], audio_data[i][1])[np.newaxis, :]) for i in range(0,batch_size)]

    train_seq_len=np.asarray([len(audio_new[s][0]) for s in range(0,batch_size)])
    
    maxlen=np.max(train_seq_len)
    
    train_inputs =np.zeros((batch_size,maxlen,num_features), np.float32)
    for i, l in enumerate(train_seq_len):
        train_inputs[i, :l, :] = audio_new[i]
    
    
    feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
    
    d = session.run(decoded[0], feed_dict=feed)
    dense_decoded = tf.sparse_tensor_to_dense(d).eval(feed_dict=feed)
    str_decoded = list(map(decode, dense_decoded))
    original=list(map(decode, text))
            
    for s in random.sample(list(zip(original, str_decoded)), 1):
        print('Original val: %s' % s[0] )
        print('Decoded val: %s' % s[1])