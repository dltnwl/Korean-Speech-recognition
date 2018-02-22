# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = 19+21+28+1+1

# Hyper-parameters
num_epochs = 20
num_hidden = 50
num_layers = 1
batch_size = 2
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 10
num_batches_per_epoch = int(num_examples/batch_size)