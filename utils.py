import torch
from ddpg_agent import Agent
import numpy as np
import matplotlib.pyplot as plt


def load_trained_agent(filepath):
    """ Load the results an parameters of a trained agent"""
    checkpoint = torch.load(filepath)
    agent = Agent(state_size=checkpoint['state_size'],
                 action_size=checkpoint['action_size'],
                 random_seed=checkpoint['seed'],
                 hidden_layers=checkpoint['hidden_layers'],
                 n_agents=checkpoint['n_agents'])
    
    agent.actor_local.load_state_dict(checkpoint['al_state_dict'])
    agent.critic_local.load_state_dict(checkpoint['cl_state_dict'])
    
    return agent

def save_results(agent,scores,file_name):
    """Save the infor and parameters to reconstruct the agent"""
    
    checkpoint = {'state_size': agent.state_size,
                  'action_size': agent.action_size,
                  'hidden_layers':agent.hidden_layers,
                  'seed':agent.seed,
                  'n_agents':agent.n_agents,
                  'scores':scores,
                  'al_state_dict': agent.actor_local.state_dict(),
                  'at_state_dict': agent.actor_target.state_dict(),
                  'cl_state_dict': agent.critic_local.state_dict(),
                  'ct_state_dict': agent.critic_target.state_dict(),            
                 }
    
    torch.save(checkpoint, './saved_models/'+file_name)
    

def show_agent_performance(scores):
    """ Show the score of obtained by an agent during his training phase"""
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='scores')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    moving_average=[np.mean(scores[t:t+100])for t in range(len(scores))]
    plt.plot(moving_average, label='100-steps moving average')
    plt.legend(loc='upper left')
    plt.show()

