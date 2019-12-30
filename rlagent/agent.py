import torch
from rlagent.memorybuffer import MemoryBuffer
import random


class Network(torch.nn.Module):
    def __init__(self,feature_cnt,output_cnt):
        super(Network,self).__init__()

        # Move model to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.fc1 = torch.nn.Linear(feature_cnt,32,bias=False)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = torch.nn.Linear(32, output_cnt, bias=False)
        self.fc2.weight.data.normal_(0, 0.1)
    
    def forward(self,x):
        out = torch.nn.ReLU()(self.fc1(x))
        out = self.fc2(out)
        return out
        

class Agent():
    def __init__(self,feature_cnt=4,action_cnt=2,memory_size=300,gamma=0.999,epsilon=1.0,epsilon_min=0.05,epsilon_decay=0.9995,lr=0.0005,ddqn_update_tempo=20):
        self.memory = MemoryBuffer(max_size=memory_size,feature_cnt=feature_cnt)
        self.base_model = Network(feature_cnt=feature_cnt,output_cnt=action_cnt)
        self.update_model = Network(feature_cnt=feature_cnt,output_cnt=action_cnt)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.update_model.parameters(),lr=lr)
        self.optimizer.zero_grad()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 200], gamma=0.1)

        self.feature_cnt=feature_cnt
        self.action_cnt=action_cnt
        
        self.gamma = gamma

        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min = epsilon_min

        self.ddqn_update_tempo = ddqn_update_tempo
        self.ddqn_countdown = self.ddqn_update_tempo

    def remember(self,state,state_prime,action,reward,isdone):
        self.memory.append(state,state_prime,action,reward,isdone)

    def act(self,state):
        if random.random()<self.epsilon:
            return random.randint(0,self.action_cnt-1)

        state = torch.from_numpy(state).float()
        action = int(torch.argmax(self.base_model(state)))
        return action
    
    def replay(self,batch_size=32,epoch_cnt=1):

        if not self.memory.is_ready(batch_size):
            return None

        s,sprime,action,reward,isdone = self.memory.sample(batch_size)

        epoch_cnt=1
        for epoch in range(epoch_cnt):
            target_q_values = reward+self.base_model(sprime).max(1).values.detach()*(~isdone)
            pred_q_values = self.update_model(s)[torch.arange(batch_size),action]

            loss = abs(target_q_values-pred_q_values).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.update_model.train()

        self.epsilon = max(self.epsilon*self.epsilon_decay,self.epsilon_min)

        if self.ddqn_countdown==0:
            self.ddqn_countdown = self.ddqn_update_tempo
            self.base_model.load_state_dict(self.update_model.state_dict())
        else:
            self.ddqn_countdown-=1