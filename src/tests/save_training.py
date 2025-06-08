from __future__ import annotations
from typing import *
from ..core.environment import *

def setup():
    map = Map({
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2]
    })
    evil_kitty = Kitty(node=2, kitty_type=Kitty.types.EVIL_KITTY)
    good_kitty = Kitty(node=1, kitty_type=Kitty.types.GOOD_KITTY)
    goal_node = 0
    initial_state = State(evil_kitty=evil_kitty, good_kitty=good_kitty, goal_node=goal_node, map=map)
    environment = Environment(initial_state)
    
    good_kitty.setup_learning_algorithm(
        learning_rate=0.01,
        discount_factor=0.9,
        epsilon=0.00
    )
    
    return environment
    
    
def show_qtable(good_kitty: Kitty):
    for index in range(64):
        i, j, k = good_kitty.qlearn_algorithm.decode_qtable_index(index)
        print(f'({i}, {j}, {k}): {good_kitty.qlearn_algorithm.qtable[index]}')


def train_kitty(environment: Environment):
    for i in range(50*environment.map.num_nodes**4):
        environment.good_kitty.act()
        environment.evil_kitty.act()
        environment.good_kitty.forward_train()


def save_kitty(good_kitty: Kitty, path: str):
    good_kitty.qlearn_algorithm.save_qtable(path)


def load_kitty(good_kitty: Kitty, path: str):
    good_kitty.qlearn_algorithm.load_qtable(path)


def test_save():
    path = 'models/good_kitty.npy'
    environment = setup()
    train_kitty(environment)
    show_qtable(environment.good_kitty)
    save_kitty(environment.good_kitty, path)


def test_load():
    path = 'models/good_kitty.npy'
    environment = setup()
    load_kitty(environment.good_kitty, path)
    show_qtable(environment.good_kitty)


def main():
    test_case = input()
    if test_case == 'load': test_load()
    if test_case == 'save': test_save()

main()