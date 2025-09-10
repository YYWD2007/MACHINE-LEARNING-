import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_epochs = 30
network = NeuralNetwork(784, 256, 256, 10)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(num_epochs):
    train_loss, train_acc = network.train_epoch_ch3(train_iter)
    test_acc = network.predict_ch3(test_iter)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

plt.clf()
plt.plot(range(1, epoch+2), train_losses, label='train loss')
plt.plot(range(1, epoch+2), train_accs, label='train acc')
plt.plot(range(1, epoch+2), test_accs, label='test acc')
plt.xlabel('epoch')
plt.ylabel('value')
plt.ylim(0, 1)
plt.legend()
plt.pause(0.1)
plt.show()

print(train_losses[-1], train_accs[-1], test_accs[-1]) 

train_loss, train_acc = network.train_epoch_ch3(train_iter)
test_acc = network.predict_ch3(test_iter)
print(f'Final train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')



