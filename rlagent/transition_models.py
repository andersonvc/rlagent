from torch import nn

class StateTransitionModel(nn.Module):
    def __init__(self,feature_cnt,action_cnt):
        super(StateTransitionModel, self).__init__()
        
        hidden_layer_cnt = 64
        
        self.fc1 = nn.Linear(feature_cnt,hidden_layer_cnt)
        self.fc2 = nn.Linear(hidden_layer_cnt,action_cnt)
    
    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x