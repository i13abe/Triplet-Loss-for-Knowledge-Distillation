from score import Score
import sys

class ScoreCalc(object):
    def __init__(self, losses, corrects, batch_size):
        self.losses = losses
        self.corrects = corrects
        
        self.batch_size = batch_size
        
        self.len_l = len(losses)
        self.len_c = len(corrects)
        
        self.train_losses = [[] for l in range(self.len_l)]
        self.train_corrects = [[] for c in range(self.len_c)]
        
        self.test_losses = [[] for l in range(self.len_l)]
        self.test_corrects = [[] for c in range(self.len_c)]
       
    
    def calc_sum(self, losses, corrects):
        if len(losses) != len(self.losses):
            print("warning : len(losses) != len(self.losses)")
            sys.exit()
        if len(corrects) != len(self.corrects):
            print("warning : len(corrects) != len(self.corrects)")
            sys.exit()
        
        for l in range(self.len_l):
            self.losses[l].sum_score(losses[l])
        
        for c in range(self.len_c):
            self.corrects[c].sum_score(corrects[c])
        
        return self.losses, self.corrects
    
    
    def score_del(self):
        for loss in self.losses:
            loss.init_score()
        for correct in self.corrects:
            correct.init_score()

        
    def score_print(self, data_num, train = True):
        if train:
            print("train mean loss={}, accuracy={}".format(self.losses[0].get_score()*self.batch_size/data_num, float(self.corrects[0].get_score()/data_num)))
        else:
            print("test mean loss={}, accuracy={}".format(self.losses[0].get_score()*self.batch_size/data_num, float(self.corrects[0].get_score()/data_num)))

            
    def score_append(self, data_num, train = True):
        if train:
            for l in range(self.len_l):
                self.train_losses[l].append(self.losses[l].get_score()*self.batch_size/data_num)
            for c in range(self.len_c):
                self.train_corrects[c].append(float(self.corrects[c].get_score()/data_num))
        else:
            for l in range(self.len_l):
                self.test_losses[l].append(self.losses[l].get_score()*self.batch_size/data_num)
            for c in range(self.len_c):
                self.test_corrects[c].append(float(self.corrects[c].get_score()/data_num))
    
    
    def get_value(self, train = True):
        if train:
            return self.train_losses, self.train_corrects
        else:
            return self.test_losses, self.test_corrects