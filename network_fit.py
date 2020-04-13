import torch
import torch.nn as nn
import numpy as np

class NetworkFit(object):
    def __init__(self, model_t, model_s, optimizer, soft_criterion, triplet_loss):
        self.model_t = model_t
        self.model_s = model_s
        self.optimizer = optimizer
        
        self.soft_criterion = soft_criterion
        self.triplet_loss = triplet_loss
        
        self.model_t.eval()
        

    def train(self, inputs, labels, kd_lambda = 2.0):
        self.optimizer.zero_grad()
        self.model_s.train()

        img1_t = inputs[0]
        img2_s = inputs[1]
        img3_s = inputs[2]
        
        label1_t = labels[0]
        label2_s = labels[1]
        label3_s = labels[2]
        
        out1_t = self.model_t(img1_t)
        out2_s = self.model_s(img2_s)
        out3_s = self.model_s(img3_s)
        
        soft_loss = self.soft_criterion(out2_s, label2_s)
        trip_loss = self.triplet_loss(out1_t, out2_s, out3_s)

        loss = soft_loss + kd_lambda*trip_loss

        loss.backward()
        self.optimizer.step()
            
            
    def test(self, inputs, labels, kd_lambda = 2.0):
        self.model_s.eval()
        
        img1_t = inputs[0]
        img2_s = inputs[1]
        img3_s = inputs[2]
        
        label1_t = labels[0]
        label2_s = labels[1]
        label3_s = labels[2]
        
        out1_t = self.model_t(img1_t)
        out2_s = self.model_s(img2_s)
        out3_s = self.model_s(img3_s)
        
        soft_loss = self.soft_criterion(out2_s, label2_s)
        trip_loss = self.triplet_loss(out1_t, out2_s, out3_s)

        loss = soft_loss + kd_lambda*trip_loss
        
        _, predicted = out2_s.max(1)
        correct = (predicted == label2_s).sum().item()
        
        return [loss.item(), soft_loss.item(), trip_loss.item()], [correct]
        
        
    
