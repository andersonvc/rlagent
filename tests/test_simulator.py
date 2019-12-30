from rlagent import __version__

from rlagent.simulator import Simulator

def test_version():
    assert __version__ == '0.1.0'

def test_simulator():
    sim = Simulator(gym_name='CartPole-v0')
    assert(sim.action_cnt==2)
    assert(sim.feature_cnt==4)

    sim = Simulator(gym_name='LunarLander-v2')
    assert(sim.action_cnt==4)
    assert(sim.feature_cnt==8)