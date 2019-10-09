import gym
import numpy as np
import blinker
import ray

from ray.rllib.agents.ppo import PPOAgent, DEFAULT_CONFIG
from ray.tune.registry import register_env

# redis_address = None
# redis_address = "localhost:6379" 
ray.init(ignore_reinit_error=True)

def make_costly_env(env_config):
    return blinker.CostlyObservations(**env_config)

register_env("CostlyObservations", make_costly_env)

config = DEFAULT_CONFIG.copy()
config['simple_optimizer'] = False
config['num_workers'] = 2
config['num_sgd_iter'] = 5
config['sgd_minibatch_size'] = 64
# config['model']['use_lstm'] = False
# config['model']['fcnet_hiddens'] = [64, 64]
config['model']['use_lstm'] = True
config['model']['lstm_cell_size'] = 32
config['model']['fcnet_hiddens'] = [5]
config['num_cpus_per_worker'] = 0

config['env'] = "CostlyObservations"

config['env_config'] = {
    'env': "CartPole-v0",
    'observation_cost': 0,
    'include_staleness': True
}

def train_agent_for(agent, epochs):
    for i in range(epochs):
        result = agent.train()
        mean_reward = result["episode_reward_mean"]
        mean_length = result["episode_len_mean"]
        print("epoch = %02d, reward = %03d, len = %03d" % (i, mean_reward, mean_length))
    return agent.save("results")

def visualize_agent(agent, name):
    env = make_costly_env(config['env_config'])
    env = gym.wrappers.Monitor(env, name, force=True)
    state = env.reset()
    cumulative_reward = 0
    policy = agent.get_policy('default')
    rnn_state = policy.get_initial_state() # for lstm-based model
    done = False
    while not done:
        action = agent.compute_action(state, state=rnn_state)
        if rnn_state:
            action, rnn_state, logits = action
        action = (action[0][0], action[1][0])
        state, reward, done, _ = env.step(action)
        env.render()
        cumulative_reward += reward
    env.close()
    print("reward =", cumulative_reward)

checkpoint_path = None

for cost in range(4):
    print("training with cost =", cost)
    config['env_config']['observation_cost'] = cost
    agent = PPOAgent(config)
    if checkpoint_path: 
        agent.restore(checkpoint_path)
    checkpoint_path = train_agent_for(agent, 10 + cost * 15)
    print("saving checkpoint to ", checkpoint_path)
    visualize_agent(agent, "results/recording_%d" % cost)
