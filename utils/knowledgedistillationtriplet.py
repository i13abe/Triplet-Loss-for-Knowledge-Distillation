import torch
from tqdm import tqdm
import numpy as np


class KnowledgeDistillationTriplet(object):
    def __init__(
        self,
        student_model,
        teacher_model,
        student_optimizer,
        soft_criterion,
        teacher_optimizer=None,
        hard_criterion=None,
        triplet_loss=None,
        lam=1.0,
        triplet_lam=1.0,
    ):
        """This is classifier fitter with knowledge distillation with Triplet.

        Args:
            student_model: Student torch model.
            teacher_model: Teacher torch model.
            student_optimizer: Optimizer of student model.
            soft_criterion: Criterion between outpus of sutudent and teacher.
            teacher_optimizer: Optimizer of teacher model. Defaults to None.
            hard_criterion: criterion between labels and student outputs.
            triplet_loss: Triplet loss.
            lam: Coefficient of hard_criterion.
            triplet_lam: Coefficient of triplet loss.
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.soft_criterion = soft_criterion
        self.hard_criterion = hard_criterion
        self.triplet_loss = triplet_loss
        self.lam = lam
        self.triplet_lam = triplet_lam
    
    
    def fit(
        self,
        EPOCH,
        trainloader,
        testloader=None,
        validation_mode=True,
        scheduler=None,
        device="cuda:0",
        lam=None,
        hard_target=True,
        teacher_optimizer=False,
        triplet_lam=None,
    ):
        losses = {"train":[], "test":[]}
        soft_losses = {"train":[], "test":[]}
        hard_losses = {"train":[], "test":[]}
        triplet_losses = {"train":[], "test":[]}
        accuracies = {"train":[], "test":[]}
        for epoch in range(EPOCH):
            print(f"epoch:{epoch+1}")
            self.train(
                trainloader,
                device,
                lam=lam,
                hard_target=hard_target,
                teacher_optimizer=teacher_optimizer,
                triplet_lam=triplet_lam,
            )
            if validation_mode:
                print("Training data results-----------------------------")
                loss, soft_loss, hard_loss, triplet_loss, acc = self.test(
                    trainloader,
                    device,
                    lam=lam,
                    hard_target=hard_target,
                    triplet_lam=triplet_lam,
                )
                losses["train"].append(loss)
                soft_losses["train"].append(soft_loss)
                hard_losses["train"].append(hard_loss)
                triplet_losses["train"].append(triplet_loss)
                accuracies["train"].append(acc)
                if testloader is not None:
                    print("Test data results---------------------------------")
                    loss, soft_loss, hard_loss, triplet_loss, acc = self.test(
                        testloader,
                        device, 
                        lam=lam,
                        hard_target=hard_target,
                        triplet_lam=triplet_lam,
                    )
                    losses["test"].append(loss)
                    soft_losses["test"].append(soft_loss)
                    hard_losses["test"].append(hard_loss)
                    triplet_losses["test"].append(triplet_loss)
                    accuracies["test"].append(acc)
            if scheduler is not None:
                scheduler.step()
        return losses, soft_losses, hard_losses, triplet_losses, accuracies
    
    
    def train(
        self,
        dataloader,
        device="cuda:0",
        lam=None,
        hard_target=True,
        teacher_optimizer=False,
        triplet_lam=None,
    ):
        device = torch.device(device)
        self.student_model.train()
        if self.teacher_optimizer is not None:
            self.teacher_model.train()
        else:
            self.teacher_model.eval()
            
        if lam is None:
            lam = self.lam
            
        if triplet_lam is None:
            triplet_lam = self.triplet_lam
            
        for (inputs, labels) in tqdm(dataloader):
            self.student_optimizer.zero_grad()
            if self.teacher_optimizer is not None:
                self.teacher_optimizer.zero_grad()

            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            
            labels[0] = labels[0].to(device)
            labels[1] = labels[1].to(device)
            
            soft_target = self.teacher_model(inputs[0])
            outputs = self.student_model(inputs[0])
            negative_outputs = self.student_model(inputs[1])
            
            soft_loss = self.soft_criterion(outputs, soft_target)

            hard_loss = 0.0
            if self.hard_criterion is not None and hard_target:
                hard_loss = self.hard_criterion(outputs, labels[0])
                
            triplet_loss = 0.0
            if self.triplet_loss is not None:
                triplet_loss = self.triplet_loss(soft_target, outputs, negative_outputs)

            loss = soft_loss + lam*hard_loss + triplet_lam*triplet_loss
            
            loss.backward()
            self.student_optimizer.step()
            if self.teacher_optimizer is not None and teacher_optimizer:
                self.teacher_optimizer.step()

                
    def test(
        self,
        dataloader,
        device="cuda:0",
        lam=None,
        hard_target=True,
        triplet_lam=None,
    ):
        sum_loss = 0.
        sum_soft_loss = 0.
        sum_hard_loss = 0.
        sum_triplet_loss = 0.
        sum_acc = 0.
        
        if lam is None:
            lam = self.lam
            
        if triplet_lam is None:
            triplet_lam = self.triplet_lam
        
        self.student_model.eval()
        self.teacher_model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            
            labels[0] = labels[0].to(device)
            labels[1] = labels[1].to(device)
            
            soft_target = self.teacher_model(inputs[0])
            outputs = self.student_model(inputs[0])
            negative_outputs = self.student_model(inputs[1])

            soft_loss = self.soft_criterion(outputs, soft_target)

            hard_loss = 0.0
            if self.hard_criterion is not None and hard_target:
                hard_loss = self.hard_criterion(outputs, labels[0])

            triplet_loss = 0.0
            if self.triplet_loss is not None:
                triplet_loss = self.triplet_loss(soft_target, outputs, negative_outputs)

            loss = soft_loss + lam*hard_loss + triplet_lam*triplet_loss
            
            sum_loss += loss.item()*len(inputs)
            sum_soft_loss += soft_loss.item()*len(inputs)
            sum_hard_loss += hard_loss.item()*len(inputs)
            sum_triplet_loss += triplet_loss.item()*len(inputs)
            _, predicted = (outputs).max(1)
            correct = (predicted == labels[0]).sum().item()
            sum_acc += correct
        
        sum_loss /= len(dataloader.dataset)
        sum_soft_loss /= len(dataloader.dataset)
        sum_hard_loss /= len(dataloader.dataset)
        sum_triplet_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        
        print(f"mean_loss={sum_loss}, mean_soft_loss={sum_soft_loss}, "\
              f"mean_hard_loss={sum_hard_loss}, triplet_loss={sum_triplet_loss}, "\
              f"acc={sum_acc}")
        
        return sum_loss, sum_soft_loss, sum_hard_loss, sum_triplet_loss, sum_acc
    
    
    def getOutputs(
        self,
        dataloader,
        based_labels=None,
        device="cuda:0",
    ):
        if isinstance(based_labels, int):
            based_labels = np.arange(based_labels)
        data_dict = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs = inputs[0].to(device)
            labels = labels[0]
            
            outputs = self.student_model(inputs)
            
            for data, label in zip(outputs, labels):
                data_dict[based_labels[label]].append(data.cpu().detach().numpy())
            
        for key in based_labels:
            data_dict[key] = np.vstack(data_dict[key])
        return data_dict

    
    def setTeacherModel(self, model):
        self.teacher_model = model
        
        
    def setStudentModel(self, model):
        self.student_model = model
    
    
    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        
        
    def setSoftCriterion(self, criterion):
        self.soft_criterion = criterion
    
    
    def setHardCriterion(self, criterion):
        self.hard_criterion = criterion
        
    