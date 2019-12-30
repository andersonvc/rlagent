__version__ = '0.1.0'

from rlagent.simulator import Simulator
from rlagent.memorybuffer import MemoryBuffer
from rlagent.agent import Agent

def cartpole():
    sim = Simulator(gym_name='CartPole-v0',max_step=199,early_termination_penalty=-200)

    feature_cnt = sim.feature_cnt
    action_cnt = sim.action_cnt
    agent = Agent(feature_cnt,action_cnt,lr=0.1,gamma=0.9999,epsilon_decay=0.995,epsilon_min=0.1,memory_size=10000,ddqn_update_tempo=5)

    sim.run(agent,episode_cnt=3000,render=True)