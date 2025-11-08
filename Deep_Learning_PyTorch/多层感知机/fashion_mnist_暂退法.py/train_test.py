from net import Net
#from dropout import dropout_layer
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs, batch_size = 50, 256
net = Net(num_inputs=784, num_hiddens1=256, num_hiddens2=256, num_outputs=10)
net.to(device)
optimizer = net.updater()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(num_epochs):
    train_loss, train_acc = net.train_epoch_ch3(train_iter, optimizer)
    test_acc = net.predict_ch3(test_iter)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
 
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='train loss')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='train acc')
plt.plot(test_accs, label='test acc')
plt.xlabel('epoch'); plt.ylabel('accuracy')
plt.ylim(0,1)
plt.legend()
plt.show()


print(train_losses[-1], train_accs[-1], test_accs[-1]) 

train_loss, train_acc = net.train_epoch_ch3(train_iter, optimizer)
test_acc = net.predict_ch3(test_iter)
print(f'Final train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')
