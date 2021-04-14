# P3: Collaboration and Competition - Tennis Match
<p align="center">
<img align="center" src="https://github.com/chuquikun/Competition_Collaboration_Project/blob/main/images/match_3.gif">
</p>
<p align="center">DDPG Trained Tennis Player Agents</p>


### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Setting the environment up

1. To run the environment we need a specific version of ptyhon so you are ancouraged to set a conda environment. Here the env was called dqn_bca (deep q network banana collector agent):
* For Linux or Mac:

```bash
conda create --name drlnd python=3.6
conda activate drlnd
```
* For Windows:
```bash
conda create --name drlnd python=3.6 
activate drlnd
```

2. Perform a minimal installation of OpenAI gym:

*  Run the following line to perform minimal installation
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
* Install the **classic control** and **box2** environments by runnning:
```
pip install -e '.[classic_control]'
pip install -e '.[box2d]'
```

3. Clone or download this repository:
```
https://github.com/chuquikun/Continuous_Control_Project-Reacher.git
```
* Move to the folder `python/ ` and install dependencies within:
```
cd python
pip install .
```
4. Create an IPython kernel for the drlnd environment:

```
python -m ipykernel install --user --name drlnd --display-name "drlnd".
```
5. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
    Alternatively you can download a headless version of the Linux environment which is very convenient to train the agent without launch the graphic interface.
    - Linux No Visualization:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    
6. Place the file in the the roor of this repository and unzip (or decompress) the file. 

7. Running the notebooks 

To launch the notebooks run in the root of this directory:
```
jupyter notebook
```
Finally select and double-click the notebook you want to run.
Before running code in any notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

### Instructions

This repository contains the next files:

- **model.py** contains the code to train the neural networks that represent the Actor and Critic for the DDPG agent.
- **ddpg_agent.py** decribes the Agent class and contains how the agents act and learn.
- **collaboration_and_competition_tennis.ipynb** this is the notebook you may want to run to train a new agents or modify the existing ones.
- **run_trained_agent.ipynb** this is the notebook that you need to run if you want to see the performance of the already trained agents.
- **utils.py** contains a utility functions to save and load trained agents and to show their performance.
- **Report.md** contains an a briefly explanation of the algorithms used to train the agent the parameters and hypermeters used and a briefly explanation of the results.
