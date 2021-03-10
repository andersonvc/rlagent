# RLAgent DDQN Reinforcement Learning Test Harness
This project implements a Double Deep Q-Network Agent (DDQN) which can be easily tuned via config parameters to solve a wide range of OpenAI-gym simulations. In addition to the agent, a simulator class is included, which serves as a wrapper around the OpenAI library to collect model weights & metrics during training & is used to profile the model's loss function; rewards, and epsilon decay during training.

To see an example of how to use this library, see CartPole_Example.ipynb which goes through the steps of training a model to solve the 'Cartpole-v1' controls problem.
 - Cartpole-v1 Solution: [Cartpole Notebook](https://github.com/andersonvc/rlagent/blob/main/CartPole_Example.ipynb)
 - Acrobot-v1 Solution: [Acrobot Notebook](https://github.com/andersonvc/rlagent/blob/main/Acrobot_Example.ipynb)

In order to use the agent, a transition model will need to be initialized and passed in as a config parameter. (In the library examples, the transition model is built using pytorch). The agent allows the user to set the following parameters:
 - memory_size: max number of state-action entries that can be stored in the memory buffer
 - gamma: discount factor (values closer to 1.0 propagate reward over a longer time horizon)
 - epsilon: starting epsilon greedy value (prob agent will take random action instead of optimal one)
 - epsilon_decay: epsilon multiplier applied te epsilon at each epoch
 - epsilon_min: lowest alowed epsilon greedy value during training (set to 0 during test)
 - lr: approx. transition model learning rate
 - batch_size: number of memory buffer values used in transition model's minibatch
 - batch_cnt: number of batches to process in each epoch
 - smoothing: when updating the base Q network after training the target network, it is updated by a smoothing weight to reduce the likelihood of model divergence: base\_model=base\_model\*smoothing+(1-smoothing)\*target\_model 
 - action_cnt: \# of discrete actions available (agent currently assumes discrete action space)
 - feature_cnt: \# of feature observations returned from the simulator (discrete or continuous)
 - weight_filepath: where to save/load trained model weights

The simulator class takes the following parameters during configuration:
- train_mode: bool to decide whether to train model (or just load pretrained weights)
- training_epochs: max \# of epochs to run training simulation
- early\_stopping\_condition: tuple of (A,B) where simulation stops early if the past A epochs all had rewards >= B
- plot_filepath: where to save plots generated during training