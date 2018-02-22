with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for per in range(num_batches_per_epoch):
            
            try :
                step(next(batch2(new)))
            except  IndexError:  
                step(next(batch2(new)))