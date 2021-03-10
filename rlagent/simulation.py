from .agent import Agent
import gym
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class Simulator:
    def __init__(self,gym_model_name:str,agent:Agent,**args):
        
        self.env = gym.make(gym_model_name)
        self.agent = agent
        self.visualize_test = args['visualize_test'] if 'visualize_test' in args else True
        self.training_epochs = args['training_epochs'] if 'training_epochs' in args else 1000
        self.train_model = args['train_model'] if 'train_model' in args else True
        self.plot_filepath = args['plot_filepath'] if 'plot_filepath' in args else 'training_plot.png'
        self.early_stopping_condition = args['early_stopping_condition'] if 'early_stopping_condition' in args else None
        self.episode_cnt = args['episode_cnt'] if 'episode_cnt' in args else 10
    
    def run(self):
        if self.train_model:
            self.terminate_early=False
            self.train_metrics = {'epsilon':[],'rewards':[],'loss':[]}
            for i in range(self.training_epochs):
                for trial in range(self.episode_cnt):
                    self.run_trial()
                
                self.agent.base_transition_network.train()
                curr_loss_mean,curr_loss_std = self.agent.update_transition_model()
                self.train_metrics['loss'].append((i,curr_loss_mean,curr_loss_std))
                self.train_metrics['epsilon'].append((i,self.agent.epsilon))
                
                if i%25==0 and i:
                    total_rewards = 0
                    trial_cnt = 10
                    self.agent.base_transition_network.eval()
                    for j in range(trial_cnt):
                        total_rewards+=self.run_trial(use_epsilon=False)
                    self.agent.base_transition_network.train()
                    self.train_metrics['rewards'].append((i,total_rewards/trial_cnt))
                    #self.train_metrics['epsilon'].append(self.agent.epsilon)
                    print(f"Epoch: {i}/{self.training_epochs}, Rewards: {self.train_metrics['rewards'][-1]}, Loss: {self.train_metrics['loss'][-1][1]},Epsilon: {self.train_metrics['epsilon'][-1][1]}, Memory: {len(self.agent.memory)}")

                if self.early_stopping_condition:
                    past_trial_cnt = self.early_stopping_condition[0]
                    min_reward = self.early_stopping_condition[1]
                    if len([reward for reward in self.train_metrics['rewards'][-1*past_trial_cnt:] if reward[1]>=min_reward])>=past_trial_cnt:
                        break

            self.agent.save_weights()
            self.plot_training_results()
        
        self.test_rewards = []
        self.agent.load_weights()
        self.agent.base_transition_network.eval()
        trial_cnt=10
        for trial in range(trial_cnt):
            self.test_rewards.append(self.run_trial(use_epsilon=False,use_visualization=self.visualize_test))
        self.env.close()
        
        print(f'Average trial score: {np.mean(self.test_rewards)}\u00B1{np.std(self.test_rewards)}')

                
    def run_trial(self,use_epsilon=True,use_visualization=False):
        
        curr_state = self.env.reset()
        if use_visualization:
            self.env.render()
        replay_records = []
        cum_reward=0
        done=False
        
        while cum_reward<2000 and not done: 
            action = self.agent.get_action(curr_state,use_epsilon)
            s_prime, r, done, _ = self.env.step(action)
            if use_visualization:
                self.env.render()
            
            # Add reward shaping for CartPole-v1
            if self.env.spec.id=='CartPole-v1':
                if done and cum_reward<499:
                    r = -100
            
            transition = {'s':curr_state,'s_prime':s_prime,'a':action,'r':r,'is_done':done}

            if use_epsilon:
                self.agent.append_replay(transition)
            cum_reward += r
            curr_state = s_prime
            
        return cum_reward
    
    def plot_training_results(self,plot_filepath=None):
        fig, axes = plt.subplots(2, 2,figsize=(20, 10))
        df = pd.DataFrame(self.train_metrics['loss'],columns=['Epoch','Smoothed Loss','Smoothed Loss Std Dev'])
        df['Smoothed Loss'] = df['Smoothed Loss'].rolling(15, win_type='gaussian').sum(std=3)
        df['Smoothed Loss Std Dev'] = df['Smoothed Loss Std Dev'].rolling(15, win_type='gaussian').sum(std=3)
        sns.lineplot(ax=axes[0,0],data=df, x='Epoch',y='Smoothed Loss')
        sns.lineplot(ax=axes[0,1],data=df,x='Epoch',y='Smoothed Loss Std Dev')
        for i in range(2): 
            for j in range(2):
                axes[i,j].set_xlim(0,df['Epoch'].max())

        axes[0,0].set_ylim(0, df['Smoothed Loss'].max())
        axes[0,1].set_ylim(0, df['Smoothed Loss'].max())

        df = pd.DataFrame(self.train_metrics['rewards'],columns=['Epoch','Reward'])
        sns.lineplot(ax=axes[1,0],data=df,x='Epoch',y='Reward')
        axes[1,0].set_ylim(df['Reward'].min()-10, df['Reward'].max()+10)
        df = pd.DataFrame(self.train_metrics['epsilon'],columns=['Epoch','Epsilon'])
        sns.lineplot(ax=axes[1,1],data=df,x='Epoch',y='Epsilon')
        axes[1,1].set_ylim(0, df['Epsilon'].max())
        fig.subplots_adjust(top=0.95)
        fig.suptitle('DDQN Model Training Weight / Metric Profiles')
        plt.tight_layout()
        if plot_filepath:
            fig.savefig(plot_filepath)
        elif self.plot_filepath:
            fig.savefig(self.plot_filepath)
        else:
            print('No plotting filepath seleted')