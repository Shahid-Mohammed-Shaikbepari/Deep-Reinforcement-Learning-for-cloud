# Deep-Reinforcement-Learning-for-cloud
Deep Q Network model for two stage resource provisioning and task scheduling for Cloud computing

#### Shahid Mohammed Shaikbepari
##### shaikbep@usc.edu
#### The list of files:
1.	DQN_skeleton.py: This has the code for Deep Q Network model and agent and all the supporting functions related to DQN
2.	Env_dqn.py: This file has the environment of cloud, user workload model for resource provisioning and task scheduling and all functions related to it
3.	Env_dqn.txt: results on DQN with 1000 - 5000 tasks on 100 - 300 servers
4.	Input.txt: the input data of user workload model from Google cluster-usage traces after extracting.
5.	Requirements.txt: List of necessary libraries required to execute the project

#### To run:
Assuming to run on Linux Ubuntu virtual machine please follow the following the steps:

```
    $ sudo pip install virtualenv
    $ virtualenv -p python3 .env # create a virtual environment
    $ source .env/bin/activate # activate the virtual environment
    $ pip install -r requirements.txt
    $ python env_dqn.py
```


#### Summary of project: 
The cloud service providers face a lot of expenses every year in terms of electricity, however, this can be minimized by intelligently utilizing the resources by resource provisioning and task scheduling. In this project, I have developed a two stage deep Q network which deep learning based which has an agent which could be trained to based on reinforcement learning to select the right resource for the current task at queue and also reduce the rejection rates. In the first stage the agent chooses the server farm and in second stage we choose server.

#### Structure of Project:
1.	Build the DQN using a PyTorch library
2.	State space: Current state of the available resources viz. server’s CPU, memory
3.	Action space: Total choices available to choose from i.e. total server farms (stage1), servers (stage2)
4.	Reward Function: Based upon the choice the DQN agent made, we feedback it with the rewards for it by calculating the it through the power it consumed in the process
5.	Experience replay: To benefit from the rare experiences we store the actions and rewards and after some time we choose from the memory buffer and relearn from it, because of which our model is not too biased even if we have same type of actions taking place
6.	Environment: This to simulate the cloud environment, a hierarchy with server farms, servers and virtual machines has been designed and allocated CPU, memory. Energy model, reward functions, rejection model are designed here.



#### Some parameters and technical details:
1.	Numpy was used to process the data
2.	The DQN is a Linear Neural network
3.	Loss function is MSE 
4.	Learning rate was 0.0001, epsilon = 1 to 0.01 to explore the action space

#### Results:
> The model achieved 300% less energy consumption and 50% less rejection rate compared with the baseline models

