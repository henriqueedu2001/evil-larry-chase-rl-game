from __future__ import annotations
from typing import *
import numpy as np

class Environment:
    """The environment of the game, containing the state S with the evil kitty
    and the good kitty agents, the run and train logic and the reward R computing
    """
    
    def __init__(self, initial_state: State):
        """Initializes the Environment with an initial state S0

        Args:
            initial_state (State): _description_
        """
        self.map = initial_state.map
        self.evil_kitty = initial_state.evil_kitty
        self.good_kitty = initial_state.good_kitty
        self.state = initial_state
        self.logs = False
    
    
    def run(self):
        """Runs the game until the evil kitty reaches the good kitty or
        the good kitty finds the goal node on the map
        """
        self.log_state()
        while not self.finished_game():
            self.step_forward()
            self.log_state()
    
    
    def step_forward(self):
        """Steps forward the simulation of the environment, by obtaining the
        actions of the evil kitty and the good kitty and computing the rewards
        """
        old_state = self.state
        self.good_kitty.act()
        self.evil_kitty.act()
        self.good_kitty.get_reward(old_state=old_state, new_state=self.state)
        self.evil_kitty.get_reward(old_state=old_state, new_state=self.state)
    
    
    def finished_game(self) -> bool:
        """Checks if the game episode is finished

        Returns:
            bool: True if the game is finished; False otherwise
        """
        if self.state.evil_kitty.node== self.state.good_kitty.node: return True
        if self.state.good_kitty.node == self.state.goal_node: return True
        return False
    
    
    def log_state(self):
        """Logs the state variables: (a) evil_kitty position, (b) good kitty position ad (c) goal
        position. It only logs if the variable logs is set True in this class
        """
        evil_kitty = self.state.evil_kitty.node
        good_kitty = self.state.good_kitty.node
        goal = self.state.goal_node
        print(f'good_kitty: {good_kitty} | evil kitty: {evil_kitty} | goal: {goal}')


class Map:
    """Represents the map of the game with a non oriented graph. Every node of the graph
    is represented by an integer i; the connections between the nodes are encoded by a
    dictionary of adjacencies 
    """
    
    def __init__(self, graph: Dict[int, List[int]]):
        """Initializes the map with the given graph, represented by a dictionary of 
        adjacencies.

        Args:
            graph (Dict[int, List[int]]): the dictionary of adjacencies of the graph
        """
        self.graph = graph
        self.num_nodes = len(self.graph)
    
    
    def get_neighbors(self, node: int) -> List[int]:
        """For a given node, obtains all his adjacent nodes (neighbors)

        Args:
            node (int): the node id

        Returns:
            List[int]: the list of all the adjacent nodes
        """
        neighbors = self.graph[node]
        return neighbors


class State:
    """Represents the state S of the game, with the evil kitty and good kitty positions, the 
    goal node position, and the map
    """
    
    def __init__(self, evil_kitty: Kitty, good_kitty: Kitty, goal_node: int, map: Map):
        """Initializes the State S with the given initial state S0.

        Args:
            evil_kitty (Kitty): the evil kitty; it can be already trained or not
            good_kitty (Kitty): the good kitty; it can be already trained or not
            goal_node (int): the goal node
            map (Map): the map of the game
        """
        self.map = map
        self.evil_kitty = evil_kitty
        self.good_kitty = good_kitty
        self.goal_node = goal_node
        self.evil_kitty.state = self
        self.good_kitty.state = self
        self.evil_kitty.map = self.map
        self.good_kitty.map = self.map


class Kitty:
    """Represents a kitty (a Reinforcement Learning agent, to be more precise)
    """
    
    class types:
        """The available types of kitty
        """
        EVIL_KITTY = 'evil_kitty'
        GOOD_KITTY = 'good_kitty'
    
    def __init__(
        self,
        node: int,
        kitty_type: str,
        map: Map = None,
        state: State = None,
        reward: float = 0.0,
        qlearn_algorithm: QLearnAlgorithm = None
    ):
        """Initializes the kitty state, with his position on the map, his kitty type and his reward. This class
        also stores the map and the game state S.

        Args:
            node (int): the kitty node
            kitty_type (str): the kitty type (evil kitty or good kitty)
            map (Map, optional): the map of the game. Defaults to None.
            state (State, optional): the game state S. Defaults to None.
            reward (float, optional): the reward. Defaults to 0.0.
        """
        self.node = node
        self.kitty_type = kitty_type
        self.map = map
        self.state = state
        self.old_state = state
        self.reward = reward
        self.epsilon = 0.00
        self.qlearn_algorithm = qlearn_algorithm
        self.train = False
        self.random_strategy = True
    
    
    def setup_learning_algorithm(self, learning_rate: float, discount_factor: float, epsilon: float):
        """Setups the learning algorithm, initializing the parameters and setting the map

        Args:
            learning_rate (float): learning rate of the q-learn algorithm (hyperparameter)
            discount_factor (float): the discount factor of the q-learn algorithm (hyperparameter)
            epsilon (float): the epsilon of the epsilon-greedy strategy
        """
        self.qlearn_algorithm = QLearnAlgorithm(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            kitty=self,
            map=self.map
        )
        self.train = True
        self.random_strategy = False
        self.epsilon = epsilon
    
    
    def act(self):
        """Takes an action A, based on the game state S
        """
        self.store_old_state()
        if self.random_strategy: self.random_act()
        else: self.epsilon_greedy_act()
    
    
    def forward_train(self):
        """Passes the experience earned from the last action taken to the learning algorithm, 
        updating the q-learn table
        """
        self.get_reward(self.old_state, self.state)
        self.qlearn_algorithm.update(self.old_state, self.state, self.reward)
    
    
    def epsilon_greedy_act(self):
        """Takes an action A with the epsilon greedy strategy. It chooses a random valid action
        with a probability epsilon or the RL algorithm recommended action with 1 - epsilon
        probability 
        """
        if np.random.rand() < self.epsilon:
            self.random_act()
        else:
            self.rl_act()
    
    
    def random_act(self):
        """Takes a random valid action A for the state S
        """
        neighbors = self.map.get_neighbors(self.node)
        choice = np.random.choice(neighbors + [self.node])
        self.node = choice
    
    
    def rl_act(self):
        """Takes an action from the reinforcement learning algorithm
        """
        choice = self.qlearn_algorithm.get_best_action(self.state)
        self.node = choice
    
    
    def get_reward(self, old_state: State, new_state: State) -> float:
        """Computes the reward R of the action A from the old state S and the new state S'

        Args:
            old_state (State): the state S before the action A
            new_state (State): the state S' after the action A

        Returns:
            float: the reward R
        """
        old_good_kitty_node = old_state.good_kitty.node
        old_evil_kitty_node = old_state.evil_kitty.node
        new_good_kitty_node = new_state.good_kitty.node
        new_evil_kitty_node = new_state.evil_kitty.node
        goal = new_state.goal_node
        reward = 0.0
        
        if self.kitty_type == Kitty.types.GOOD_KITTY:
            if new_good_kitty_node == new_evil_kitty_node: reward =  -10.0
            elif new_good_kitty_node == goal: reward =  10.0
            elif old_good_kitty_node == new_good_kitty_node: reward =  -2.0
            else: reward =  -1.0
        elif self.kitty_type == Kitty.types.EVIL_KITTY:
            if new_good_kitty_node == new_evil_kitty_node: reward =  +10.0
            elif new_good_kitty_node == goal: reward =  -10.0
            elif old_evil_kitty_node == new_evil_kitty_node: reward =  +2.0
            else: reward =  -1.0
        
        self.reward = reward
    
    
    def store_old_state(self):
        """Makes an copy of the actual state, before a certain action is taken, then saves it
        on the old_state variable of the Kitty class
        """
        self.old_state = State(
            evil_kitty=Kitty(node=self.state.evil_kitty.node, kitty_type=Kitty.types.EVIL_KITTY),
            good_kitty=Kitty(node=self.state.good_kitty.node, kitty_type=Kitty.types.GOOD_KITTY),
            goal_node=self.state.goal_node,
            map=None
        )
        

class QLearnAlgorithm:
    """Represents the q-learn algorithm
    """
    def __init__(self, learning_rate: float = 0, discount_factor: float = 0, kitty: Kitty = None, map: Map = None):
        """_summary_

        Args:
            learning_rate (float, optional): learning rate of the q-learn algorithm (hyperparameter). Defaults to 0.
            discount_factor (float, optional): the discount factor of the q-learn algorithm (hyperparameter). Defaults to 0.
            kitty (Kitty, optional): the kitty that owns the algorithm and its values. Defaults to None.
            map (Map, optional): the map of the game. Defaults to None.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.kitty = kitty
        self.map = map
        self.num_nodes = self.map.num_nodes
        self.num_states = self.map.num_nodes * self.map.num_nodes
        self.num_actions = self.map.num_nodes
        self.qtable = None
        self.init_qtable()
    
    
    def init_qtable(self):
        """Initializes que q-learn table with zeros
        """
        n = self.num_nodes
        self.qtable = np.zeros(shape=(n**3, n))
    
    
    def save_qtable(self, path: str):
        """Saves the q-table as a .npy file

        Args:
            path (str): the saving path (not Jesus)
        """
        np.save(path, self.qtable)
    
    
    def load_qtable(self, path: str):
        """Load q-table from a .npy file

        Args:
            path (str): the loading path
        """
        self.qtable = np.load(path)
    
    
    def get_best_action(self, state: State) -> int:
        """Computes the best action a, on the current state S, from the q-learn algorithm

        Args:
            state (State): the current state S of the game

        Returns:
            int: the best next action, corresponding to the best next node to move
        """
        best_action = 0
        # state variables
        evil_kitty_node = state.evil_kitty.node
        good_kitty_node = state.good_kitty.node
        goal_node = state.goal_node
        my_node = self.kitty.node
        # consulting the best action on the q-table, considering the map restrictions
        qtable_index = self.get_qtable_index(evil_kitty_node, good_kitty_node, goal_node)
        qtable_row = self.qtable[qtable_index]
        allowed_actions = self.map.graph[my_node]
        best_action, _ = self.get_best_allowed_action(qtable_row, allowed_actions)
        
        return best_action
    
    
    def get_best_allowed_action(self, qtable_row: List[int], allowed_actions: List[int]) -> Tuple[int, int]:
        """From a given state S, computes the best 

        Args:
            qtable_row (List[int]): the q-table row of the current state S
            allowed_actions (List[int]): the set of the allowed actions of the state S

        Returns:
            Tuple[int, int]: the best next node to go
        """
        allowed_values = [(idx, qtable_row[idx]) for idx in allowed_actions]
        best_idx, best_value = max(allowed_values, key=lambda x: x[1])
        
        return best_idx, best_value
    
    
    def update(self, old_state: State, new_state: State, reward: float):
        """Updates the q-learn table, based on the old_state S, the new state S',
        the reward acquired by the kitty, and the action taken by the kitty from S to S' 

        Args:
            old_state (State): the state S before the action
            new_state (State): the state S after the action
            reward (float): the reward acquired
        """
        # old state variables
        old_evil_kitty_node = old_state.evil_kitty.node
        old_good_kitty_node = old_state.good_kitty.node
        old_goal_node = old_state.goal_node
        old_qtable_row_index = self.get_qtable_index(old_evil_kitty_node, old_good_kitty_node, old_goal_node)
        
        # new state variables
        new_evil_kitty_node = new_state.evil_kitty.node
        new_good_kitty_node = new_state.good_kitty.node
        new_goal_node = new_state.goal_node
        new_qtable_row_index = self.get_qtable_index(new_evil_kitty_node, new_good_kitty_node, new_goal_node)        
        
        s_old_index, s_new_index = old_qtable_row_index, new_qtable_row_index
        
        my_kitty_type = self.kitty.kitty_type
        my_node = new_state.evil_kitty.node if my_kitty_type == Kitty.types.EVIL_KITTY else new_state.good_kitty.node
        my_action = my_node
        
        # acessing the old q-value Q(s, a)
        old_qtable_row = self.qtable[s_old_index]
        new_qtable_row = self.qtable[s_new_index]
        allowed_actions = self.map.graph[my_node]
        _, greatest_qvalue = self.get_best_allowed_action(new_qtable_row, allowed_actions)
        old_qvalue = old_qtable_row[my_action]
        
        # updating the q-value Q(s, a)
        r = reward
        eta = self.learning_rate
        gamma = self.discount_factor
        new_qvalue = (1 - eta) * old_qvalue + eta * (r + gamma * greatest_qvalue)
        self.qtable[s_old_index][my_action] = new_qvalue
    
    
    def get_qtable_index(self, evil_kitty_node: int, good_kitty_node: int, goal_node: int) -> int:
        """Computes the row index l on the q-table, for a given list of indexes (i, j, k), where
        i = evil kitty node, j = good kitty node, k = goal node. The formula is given by:
        l = i + jn + kn^2

        Args:
            evil_kitty_node (int): evil kitty node
            good_kitty_node (int): good kitty node
            goal_node (int): goal node

        Returns:
            int: the l index, corresponding to the index of the row for (i, j, k)
        """
        i, j, k, n = evil_kitty_node, good_kitty_node, goal_node, self.num_nodes
        qtable_index = i + j * n + n * n * k
        return qtable_index
    
    
    def decode_qtable_index(self, qtable_index: int) -> Tuple[int, int, int]:
        """Decodes an q-table index (l) to its individual values (i, j, k), where i = evil kitty node,
        j = good kitty node, k = goal node

        Args:
            qtable_index (int): the index l

        Returns:
            Tuple[int, int, int]: (i, j, k)
        """
        n = self.num_nodes
        k = qtable_index // (n * n)
        remainder = qtable_index % (n * n)
        j = remainder // n
        i = remainder % n
        return (i, j, k)