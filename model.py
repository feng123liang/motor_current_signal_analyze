from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_time = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=9,padding=4), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
            nn.Conv2d(5,5,kernel_size=2,stride=2), # (40,40)
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
        )
        self.model_corp = nn.Sequential(
            nn.Conv2d(in_channels=5,out_channels=10,kernel_size=7,padding = 3),
            nn.BatchNorm2d(10,1e-6),
            nn.ReLU(),
            nn.Conv2d(10,10,kernel_size=2,stride=2),
            nn.BatchNorm2d(10,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,out_channels=15,kernel_size=5,padding = 2),
            nn.BatchNorm2d(15,1e-6),
            nn.ReLU(),
            nn.Conv2d(15,15,kernel_size=2,stride=2),
            nn.BatchNorm2d(15,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding = 1),
            nn.BatchNorm2d(30,1e-6),
            nn.ReLU(),
            nn.Conv2d(30,30,kernel_size=2,stride=2),
            nn.Flatten(),
            # nn.Linear(750,750),
            # nn.ReLU(),
            nn.Linear(750,3),
            nn.Softmax(dim = 1)
        )
        self.model_fre = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=6), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=5,out_channels=5,kernel_size=6), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
        )
      

    def forward(self, input_tot):
        # print(input,input.shape)
        # input_time = input_tot[:,:6400]
        # input_fre = input_tot[:,6400:]
        input_time = input_tot.reshape(-1,1,80,80) 
        input_time = self.model_time(input_time)

        # input_fre = input_fre.reshape(-1,1,50,50)
        # input_fre = self.model_fre(input_fre)

        # input_corp = torch.cat((input_time,input_fre),dim=1)
        input_corp = self.model_corp(input_time)
        # print(x.shape)
        
        return input_corp
    
def apply_model(model,data):
    
    model.eval()
    output = model(data)

    # print(output)
    pre_lab = torch.argmax(output,dim = 1)
    # print(pre_lab)

    return pre_lab,output