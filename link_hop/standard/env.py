from link_hop.util import create_graph, get_new_route, compute_reward, get_max_neighbors
from helper.graph import get_neighbors
from gym.spaces import MultiDiscrete, Discrete
from networkx import Graph
import gym
import pandas as pd

class Env(gym.Env):
    # Constructor, create graphs, set some variables for gym, house keeping stuff
    def __init__(self, save_file: str, graph: Graph = None) -> None:

        # Create our graphs, each with a unique set of edge weights
        if graph is None:
            self.graph: Graph = create_graph()
        else:
            self.graph = graph

        # Declare the spaces for the Env
        self.max_neighbors = get_max_neighbors(self.graph)

        self.observation_space: MultiDiscrete = MultiDiscrete(
            [self.num_nodes(), self.num_nodes()])
        self.action_space: Discrete = Discrete(self.max_neighbors)
        self.valid_actions = [1]*self.max_neighbors

        # Counters
        self.steps: int = 0
        self.hops: int = 0

        # Log dir
        self.save_file: str = save_file

        # Path information
        self.source: int = -1
        self.target: int = -1
        self.current_node: int = -1
        self.path: list = []
        self.neighbors = []
        self.eps = 0

        self.episode_reward = 0
        self.latency_reward = 0
        self.bandwidth_reward = 0

        f = open(self.save_file, "w+")
        f.write("steps,reward,latency,bandwidth\n")
        f.close()

        f = open("training_data/step_data.csv", "w+")
        f.write("steps,reward\n")
        f.close()

        self.df = pd.DataFrame(
            columns=['steps', 'reward'])
        self.dict_list = []

    # Preform the action and compute reward

    def step(self, action: int) -> ((int, int), float, bool, dict):
        rewards = []
        try:
            next_node = self.neighbors[action]
        except:
            return [self.current_node, self.target], -1, False, {'rewards': [-1]}
        self.path.append(next_node)
        self.current_node = next_node
        self.steps += 1
        rewards, done = self._get_reward()
        self.episode_reward += round(rewards[0], ndigits=3)
        # print(rewards[0])
        df2 = {'steps': self.steps, 'reward': round(rewards[0], ndigits=3)}
        self.dict_list.append(df2)
        if(self.steps % 1_000 == 0):
            self.df = pd.DataFrame.from_dict(self.dict_list)
            self.df.to_csv("training_data/step_data.csv", index=False,
                           header=False, mode="a")
            self.dict_list.clear()
        if done:
            with open(self.save_file, 'a') as fd:
                fd.write(str(self.steps))
                fd.write(',' + str(round(self.episode_reward, ndigits=3)))
                fd.write(',' + str(round(rewards[1], ndigits=3)))
                fd.write(',' + str(round(rewards[2], ndigits=8)))
                fd.write('\n')
                fd.close()

        self.neighbors = list(self.graph.neighbors(self.current_node))
        return [self.current_node, self.target], rewards[0], done, {'rewards': [rewards[0]]}

    # Called when an environment is finished, creates a new "environment"
    def reset(self) -> (int, int):
        return self._reset()

    # Reset counters to zero, get an observation store it and return it
    def _reset(self) -> (int, int):
        self.source, self.target = get_new_route(self.graph)
        self.current_node = self.source
        self.path = []
        self.neighbors = get_neighbors(self.graph, self.current_node)
        self.path.append(self.source)
        self.hops = 0
        self.episode_reward = 0

        return self.source, self.target

    # not used / doesn't make sense to use given the problem
    def _render(self, mode: str = 'human', close: bool = False) -> None:
        pass

    # compute the reward
    def _get_reward(self) -> (list, bool):
        return compute_reward(self.graph, self.target, tuple(self.path))

    def num_nodes(self) -> int:
        return len(self.graph.nodes)

