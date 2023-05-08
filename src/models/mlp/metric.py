import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        
        correct = 0
        raise NotImplementedError
        return correct / len(target)
    
    def __str__(self):
        return f"top{self.k}"