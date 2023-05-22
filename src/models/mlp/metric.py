import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        
        correct = 0
        ####################################################
        # TODO 
        # given an output of shape (N, C) and target labels of shape (N),
        # Compute the topK accuracy, where a "correct" classification is considered
        # when the target can be found in top-K (e.g. Top-1 or Top-5) classes.
        # Top-1 would be what's often referred to as "Accuracy".
        ####################################################
        
        _, indices = torch.topk(output, self.k, dim=1)
        for i in range(len(target)):
            if target[i].item() in indices[i]:
                correct += 1

        return correct / len(target)


    
    def __str__(self):
        return f"top{self.k}"