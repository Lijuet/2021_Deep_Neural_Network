#%%
import time
import numpy as np
from utils import load_mnist, load_fashion_mnist
from Answer import ClassifierModel, FCLayer, SoftmaxLayer, ReLU, Sigmoid, Tanh
import matplotlib.pyplot as plt
import copy

np.random.seed(123)
np.tanh = lambda x: x

model = ClassifierModel()

# =============================== EDIT HERE ===============================

"""
    Build model Architecture and do experiment.
    You can add layer as below examples.
    NOTE THAT, layers are executed in the order that you add.
    Enjoy.

    < Add Layers examples >
    - FC Layer
        model.add_layer('FC Example Layer', FCLayer(input_dim=1234, output_dim=123))
    - Softmax Layer
        model.add_layer('Softmax Layer', SoftmaxLayer())

    FC layers are usually followed by activation layers, unless it is the last FC layer.
    The last layer must be SoftmaxLayer if the task is multi-class classification(MNIST, FashionMNIST).
"""

# mnist / fashion_mnist
dataset = 'mnist'

# Hyper-parameters
num_epochs = 5
learning_rate = 0.01
reg_lambda = 1e-8
print_every = 10

batch_size = 1000

# Add layers - example 1
# model.add_layer('FC-1', FCLayer(784, 10))
# model.add_layer('Sigmoid', Sigmoid())
# model.add_layer('Softmax Layer', SoftmaxLayer())

# Add layers - example 2
model.add_layer('FC-1', FCLayer(784, 500)) # BC * 784 => BC * 500
model.add_layer('ReLU1', ReLU())
model.add_layer('FC-2', FCLayer(500, 500)) # BC * 500 => BC * 10 ( num of category )
model.add_layer('ReLU2', ReLU())
model.add_layer('FC-3', FCLayer(500, 10)) # BC * 500 => BC * 10 ( num of category )
model.add_layer('Tanh', Tanh())
model.add_layer('Softmax Layer', SoftmaxLayer()) # output is probability
# =========================================================================
assert dataset in ['mnist', 'fashion_mnist'] # give error if dataset doesn't match

# Dataset
# MNIST
# x : (5000, 1, 28, 28)(num of total dataset, reshape, x row, y col)
# y : (5000, 10)(num of total dataset, num of classes)

# Fashion_MIST
# x : (6000, 1, 28, 28)(num of total dataset, reshape, x row, y col)
# y : (6000, 10)(num of total dataset, num of classes)
if dataset == 'mnist':
    x_train, y_train, x_test, y_test = load_mnist('./data') 
else:
    x_train, y_train, x_test, y_test = load_fashion_mnist('./data') # (6000, 1, 28, 28)(num of total dataset, reshape, x row, y col)

x_train, x_test = np.squeeze(x_train), np.squeeze(x_test) # remove reshape => (# of data 5000/6000, 28, 28)

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train) # sort idx randomly
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid] # last 10% idx are valid set
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx] # split valid set
x_train, y_train = x_train[train_idx], y_train[train_idx] # split train set

num_train, height, width = x_train.shape
num_class = y_train.shape[1]

model.summary()
time.sleep(1)

train_accuracy = []
valid_accuracy = []

best_epoch = -1
best_acc = -1
best_model = None

print('Training Starts...')
num_batch = int(np.ceil(num_train / batch_size))

for epoch in range(1, num_epochs + 1):
    # model Train
    start = time.time()
    epoch_loss = 0.0
    for b in range(num_batch):
        # set start index(s) and end index(e)
        s = b * batch_size
        e = (b+1) * batch_size if (b+1) * batch_size < len(x_train) else len(x_train)

        x_batch = x_train[s:e]
        y_batch = y_train[s:e]

        loss = model.forward(x_batch, y_batch, reg_lambda)
        epoch_loss += loss

        model.backward(reg_lambda)
        model.update(learning_rate)

        print('[%4d / %4d]\t batch loss : %.4f' % (s, num_train, loss))

    end = time.time()
    lapse_time = end - start
    print('Epoch %d took %.2f seconds\n' % (epoch, lapse_time))

    if True:#epoch % print_every == 0:
        # TRAIN ACCURACY
        prob = model.predict(x_train)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_train, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        train_acc = correct / total
        train_accuracy.append(train_acc)

        # VAL ACCURACY
        prob = model.predict(x_valid)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_valid, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        valid_acc = correct / total
        valid_accuracy.append(valid_acc)

        print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))
        print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)

        if best_acc < valid_acc:
            print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
            best_acc = valid_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

print('Training Finished...!!')
print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

# TEST ACCURACY
prob = best_model.predict(x_test)
pred = np.argmax(prob, -1).astype(int)
true = np.argmax(y_test, -1).astype(int)

correct = len(np.where(pred == true)[0])
total = len(true)
test_acc = correct / total

print('Test Accuracy at Best Epoch : %.2f' % (test_acc))


"""
    Draw a plot of train/valid accuracy.
    X-axis : Epoch
    Y-axis : train_accuracy & valid_accuracy
    Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
"""
epochs = list(np.arange(1, num_epochs+1, print_every))

plt.plot(epochs, train_accuracy, label='Train Acc.')
plt.plot(epochs, valid_accuracy, label='Valid Acc.')

plt.title('Epoch - Train/Valid Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
