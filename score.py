class Score(object):
    def __init__(self, score = 0):
        self.score = score
        
    def sum_score(self, score):
        self.score += score
    
    def set_score(self, score):
        self.score = score
    
    def init_score(self):
        self.score = 0
    
    def get_score(self):
        return self.score