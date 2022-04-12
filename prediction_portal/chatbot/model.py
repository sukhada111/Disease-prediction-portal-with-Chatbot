import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

#new model
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.l3 = nn.Linear(hidden_size, hidden_size)
#         self.l4 = nn.Linear(hidden_size, hidden_size)
#         self.l5 = nn.Linear(hidden_size, hidden_size)
#         self.l6 = nn.Linear(hidden_size, num_classes)
#         self.softmax = nn.Softmax()
   
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.l2(dim=self.softmax(out))
#         out = self.l3(dim=self.softmax(out))
#         out = self.l4(dim=self.softmax(out))
#         out = self.l5(dim=self.softmax(out))
#         out = self.l6(dim=self.softmax(out))
#         # no activation and no softmax at the end
#         return out
