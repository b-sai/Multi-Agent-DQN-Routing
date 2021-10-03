# Packet Routing Using Multi Agent DQN

Sai Shreyas Bhavanasi, Dr. Flavio Esposito

The paper aims to explore Packet Routing using Deep Q Network(DQN) with a multi agent strategy. The paper assesses the performance of the algorithm compared with traditional Routing Protocols such as OSPF and ECMP on different networks topologies. The current results suggest that the Multi Agent DQN model is able to deliver packets lower delivery times than OSPF and ECMP in certain network conditions

To run the program the following command can be executed:

`python train.py`

To install the required dependencies, the following command can be run:

`sudo pip install -r requirements.txt`

The folder:

- "link_hop" contains the gym environment and the associated utility functions. The gym environment can be used for other routing algorithms as well.

- "multi_agent_DQN" contains the code for the Multi Agent DQN model.

- "helper" contains supplementary code and utility functions that can be used by various algorithms.

Additionally, the model outputs the training data in the "training_data" folder. It also saves the associated reward taken at each timestep in a file labeled "step_data".
