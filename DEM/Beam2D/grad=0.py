import torch 
import numpy as np
class mul(torch.nn.Module):
    def __init__(self, layer):
        super(mul, self).__init__()
        self.linear1 = torch.nn.Linear(layer[0], layer[1])
        self.linear2 = torch.nn.Linear(layer[1], layer[2])
        self.linear3 = torch.nn.Linear(layer[2], layer[3])
        self.linear4 = torch.nn.Linear(layer[3], layer[4])
    def forward(self, x):
         # y1 = torch.tanh(self.linear1(x))
         # y2 = torch.tanh(self.linear2(y1))
         # y3 = torch.tanh(self.linear3(y2))
         # y4 = self.linear4(y3)
         return y4
     
x = torch.ones(1,3)
y = torch.ones(3,1)*2
layer = [3, 10, 10, 10, 3]
model = mul(layer)
epoch = 100
optimizer = torch.optim.Adam(model.parameters())
for i in range(epoch):
    pred = model(x)
    loss = torch.norm(pred-y)
    loss.backward(retain_graph=True, create_graph=True)
    loss_grad = 0
    for name, param in model.named_parameters():
        loss_grad = loss_grad + torch.norm(param.grad)
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss_grad.backward()
    optimizer.update()
    