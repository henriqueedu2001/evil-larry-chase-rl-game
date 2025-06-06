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
        return
    
    
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
        return
    
    
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
        return


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
        reward: int = 0
    ):
        """Initializes the kitty state, with his position on the map, his kitty type and his reward. This class
        also stores the map and the game state S.

        Args:
            node (int): the kitty node
            kitty_type (str): the kitty type (evil kitty or good kitty)
            map (Map, optional): the map of the game. Defaults to None.
            state (State, optional): the game state S. Defaults to None.
            reward (int, optional): the reward. Defaults to 0.
        """
        self.node = node
        self.kitty_type = kitty_type
        self.map = map
        self.state = state
        self.reward = reward
        self.epsilon = 0.05
        return
    
    
    def act(self):
        """Takes an action A, based on the game state S
        """
        self.epsilon_greedy_act()
    
    
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
        # TODO implement
        neighbors = self.map.get_neighbors(self.node)
        choice = np.random.choice(neighbors)
        self.node = choice
    
    
    def get_reward(self, old_state: State, new_state: State) -> int:
        """Computes the reward R of the action A from the old state S and the new state S'

        Args:
            old_state (State): the state S before the action A
            new_state (State): the state S' after the action A

        Returns:
            int: the reward R
        """
        if self.kitty_type == Kitty.types.GOOD_KITTY:
            if self.node == new_state.goal_node: return 10
            if self.node == new_state.evil_kitty.node: return -10
            if old_state.evil_kitty.node == new_state.evil_kitty.node: return -2
            return -1
        elif self.kitty_type == Kitty.types.EVIL_KITTY:
            if self.node == new_state.goal_node: return -10
            if self.node == new_state.evil_kitty.node: return +10
            if old_state.good_kitty.node == new_state.good_kitty.node: return -2
            return -1
        return 0