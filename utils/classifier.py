import torch
from tqdm import tqdm
import numpy as np



class Classifier(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
    ):
        """This is classifier fitter.

        Args:
            model: torch model.
            optimzier: torch optimzier.
            criterion: classfier criterion.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    
    def fit(
        self,
        EPOCH,
        trainloader,
        testloader=None,
        validation_mode=True,
        scheduler=None,
        device="cuda:0",
    ):
        losses = {"train":[], "test":[]}
        accuracies = {"train":[], "test":[]}
        for epoch in range(EPOCH):
            print(f"epoch:{epoch+1}")
            self.train(trainloader, device)
            if validation_mode:
                print("Training data results-----------------------------")
                loss, acc = self.test(trainloader, device)
                losses["train"].append(loss)
                accuracies["train"].append(acc)
                if testloader is not None:
                    print("Test data results---------------------------------")
                    loss, acc = self.test(testloader, device)
                    losses["test"].append(loss)
                    accuracies["test"].append(acc)
            if scheduler is not None:
                scheduler.step()
        return losses, accuracies
            
        
    def train(
        self,
        dataloader,
        device="cuda:0",
    ):
        device = torch.device(device)
        self.model.train()
        for (inputs, labels) in tqdm(dataloader):
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            
    def test(
        self,
        dataloader,
        device="cuda:0",
    ):
        device = torch.device(device)
        sum_loss = 0.
        sum_acc = 0.
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            sum_loss += loss.item()*inputs.shape[0]
            _, predicted = self.predict(outputs)
            correct = (predicted == labels).sum().item()
            sum_acc += correct
        
        sum_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        
        print(f"mean_loss={sum_loss}, acc={sum_acc}")
        
        return sum_loss, sum_acc
    
    
    def predict(self, outputs):
        return (outputs).max(1)
    
    
    def getOutputs(
        self,
        dataloader,
        based_labels=None,
        device="cuda:0",
    ):
        assert based_labels is not None,\
        'based_labels is None. You set based_labels=[0,1,2,3,...,num_classes] or based_labels=num_classes'
        if isinstance(based_labels, int):
            based_labels = np.arange(based_labels)
        data_dict = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)
            
            for data, label in zip(outputs, labels):
                data_dict[based_labels[label]].append(data.cpu().detach().numpy())
            
        for key in based_labels:
            data_dict[key] = np.vstack(data_dict[key])
        return data_dict
    
    
    def testSummary(
        self,
        dataloader,
        device="cuda:0",
    ):
        sum_loss = 0.
        sum_acc = 0.
        
        num_classes = len(dataloader.dataset.classes)
        num_class = np.zeros((1, num_classes))[0]
        class_acc = np.zeros((1, num_classes))[0]
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            sum_loss += loss.item()*inputs.shape[0]
            _, predicted = self.predict(outputs)
            correct = (predicted == labels).sum().item()
            sum_acc += correct
            
            for i, label in enumerate(labels):
                num_class[label] += 1
                class_acc[label] += (predicted[i] == label).item()
        
        sum_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        class_acc /= num_class
        
        print(f"mean_loss={sum_loss}, acc={sum_acc}")
        
        for i in range(len(class_acc)):
            print(f"class {i} accuracy={class_acc[i]}")
        
        return sum_loss, sum_acc, class_acc
                    
        
    def setModel(self, model):
        self.model = model
    
    
    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        
        
    def setCriterion(self, criterion):
        self.criterion = criterion
        
    