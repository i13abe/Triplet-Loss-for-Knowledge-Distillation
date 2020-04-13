# to load dataset
from datasets import Datasets
from kd_triplet_datasets import KDTripletDataset
# for network and training
from network import Net_teacher, Net_student
from network_fit import NetworkFit
# to calculate the score
import savescore
from score import Score
from score_calc import ScoreCalc
# pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt


# initialize for  each parameters
DATASET = 'CIFAR10'
BATCH_SIZE = 100
NUM_WORKERS = 2
WEIGHT_DECAY = 0.007
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SCHEDULER_STEPS = 100
SCHEDULER_GAMMA = 0.1
SEED = 1
EPOCH = 500
KD_LAMBDA = 2.0
TRIPLET_MARGINE = 5.0


# fixing the seed
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# check if gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("gpu mode")
else:
    device = torch.device("cpu")
    print("cpu mode")


# the name of results files
codename = 'kd_example'
fnnname = codename + "_fnn_model"
total_loss_name = codename + "_total_loss"
soft_loss_name = codename + "_soft_loss"
tri_loss_name = codename + "_tri_loss"
acc_name = codename + "_accuracy"
result_name = codename + "_result"


# load the data set
instance_datasets = Datasets(DATASET, BATCH_SIZE, NUM_WORKERS, shuffle = False)
data_sets = instance_datasets.create()

#trainloader = data_sets[0]
#testloader = data_sets[1]
classes = data_sets[2]
based_labels = data_sets[3]
trainset = data_sets[4]
testset = data_sets[5]


# use the KD Triplet Dataset by using above dataset
tri_trainset = KDTripletDataset(trainset)
tri_testset = KDTripletDataset(testset)
tri_trainloader = torch.utils.data.DataLoader(tri_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
tri_testloader = torch.utils.data.DataLoader(tri_testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)


# network and criterions
model_t = Net_teacher().to(device)
model_s = Net_student().to(device)
model_t.load_state_dict(torch.load("cnn_alex.pkl"))

optimizer = optim.SGD(model_s.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEPS, gamma=SCHEDULER_GAMMA)

soft_criterion = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=TRIPLET_MARGINE)


# fit for training and test
fit = NetworkFit(model_t, model_s, optimizer, soft_criterion, triplet_loss)


# to manage all scores
loss = Score()
loss_s = Score()
loss_t = Score()
correct = Score()
score_loss = [loss, loss_s, loss_t]
score_correct = [correct]
sc = ScoreCalc(score_loss, score_correct, BATCH_SIZE)


# training and test
for epoch in range(EPOCH):
    print('epoch', epoch+1)
    
    for (inputs, labels) in tri_trainloader:
        img1_t = inputs[0].to(device)
        img2_s = inputs[0].to(device)
        img3_s = inputs[1].to(device)
        
        images = (img1_t, img2_s, img3_s)
        
        label1_t = labels[0].to(device)
        label2_s = labels[0].to(device)
        label3_s = labels[1].to(device)
        
        label = (label1_t, label2_s, label3_s)
        
        fit.train(images, label, KD_LAMBDA)
        
    for (inputs, labels) in tri_trainloader:
        img1_t = inputs[0].to(device)
        img2_s = inputs[0].to(device)
        img3_s = inputs[1].to(device)
        
        images = (img1_t, img2_s, img3_s)
        
        label1_t = labels[0].to(device)
        label2_s = labels[0].to(device)
        label3_s = labels[1].to(device)
        
        label = (label1_t, label2_s, label3_s)
        
        losses, corrects = fit.test(images, label, KD_LAMBDA)
        
        sc.calc_sum(losses, corrects)
    
    sc.score_print(len(trainset))
    sc.score_append(len(trainset))
    sc.score_del()
    
    for (inputs, labels) in tri_testloader:
        img1_t = inputs[0].to(device)
        img2_s = inputs[0].to(device)
        img3_s = inputs[1].to(device)
        
        images = (img1_t, img2_s, img3_s)
        
        label1_t = labels[0].to(device)
        label2_s = labels[0].to(device)
        label3_s = labels[1].to(device)
        
        label = (label1_t, label2_s, label3_s)
        
        losses, corrects = fit.test(images, label, KD_LAMBDA)
        
        sc.calc_sum(losses, corrects)
    
    sc.score_print(len(testset), train = False)
    sc.score_append(len(testset), train = False)
    sc.score_del()
    
    scheduler.step()


# get the scores
train_losses, train_corrects = sc.get_value()
test_losses, test_corrects = sc.get_value(train = False)


# output the glaphs of the scores
torch.save(model_s.state_dict(), fnnname + '.pth')
savescore.plot_score(EPOCH, train_losses[0], test_losses[0], y_lim = 5.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'total loss', filename = total_loss_name)
savescore.plot_score(EPOCH, train_losses[1], test_losses[1], y_lim = 5.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'softmax loss', filename = soft_loss_name)
savescore.plot_score(EPOCH, train_losses[2], test_losses[2], y_lim = 5.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'triplet loss', filename = tri_loss_name)
savescore.plot_score(EPOCH, train_corrects[0], test_corrects[0], y_lim = 1, y_label = 'ACCURACY', legend = ['train acc', 'test acc'], title = 'accuracy', filename = acc_name)
savescore.save_data(train_losses[0], test_losses[0], train_corrects[0], test_corrects[0], result_name)
