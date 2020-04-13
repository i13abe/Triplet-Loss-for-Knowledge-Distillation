import numpy as np
import matplotlib.pyplot as plt
    

def plot_score(epoch, train_data, test_data, x_lim = None, y_lim = None, x_label = 'EPOCH', y_label = 'score', title = 'score', legend = ['train', 'test'], filename = 'test'):
    plt.figure(figsize=(6,6))
    
    if x_lim is None:
        x_lim = epoch
    if y_lim is None:
        y_lim = 1
        
    plt.plot(range(epoch), train_data)
    plt.plot(range(epoch), test_data, c='#00ff00')
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend)
    plt.title(title)
    plt.savefig(filename+'.png')
    plt.close()

    
def save_data(train_loss, test_loss, train_acc, test_acc, filename):
    with open(filename + '.txt', mode='w') as f:
        f.write("train mean loss={}\n".format(train_loss[-1]))
        f.write("test  mean loss={}\n".format(test_loss[-1]))
        f.write("train accuracy={}\n".format(train_acc[-1]))
        f.write("test  accuracy={}\n".format(test_acc[-1]))
