import gym
import numpy as np

class Simulator():
    def __init__(self,gym_name='CartPole-v0',max_step=0,early_termination_penalty=0):
        self.env = gym.make(gym_name)
        self.max_step = max_step
        self.early_termination_penalty = early_termination_penalty

        self.feature_cnt = len(self.env.reset())
        self.action_cnt = int(self.env.action_space.n)
    

    def run(self, agent, render=False,episode_cnt=300):
        episode_reward = []
        for episode in range(episode_cnt):
            state = self.env.reset()
            done=False
            step_cnt,total_reward = 0,0.0

            while not done and (self.max_step==0 or step_cnt<self.max_step):
                
                if render and episode>400 and episode%20==0:
                    self.env.render()
                action = agent.act(state)
                state_prime, reward, done, _ =  self.env.step(action)

                if done:
                    reward = reward+self.early_termination_penalty

                agent.remember(state,state_prime,action,reward,done)

                state = state_prime
                step_cnt+=1
                total_reward+=reward
            episode_reward.append(total_reward)
            agent.replay()

            if episode % 10 == 0 and episode_reward:
                print(f'episode: {episode},avg award: {np.average(episode_reward[:-10]):.2f},\tepsilon: {agent.epsilon:.2f}')

if __name__ == "__main__":
    pass