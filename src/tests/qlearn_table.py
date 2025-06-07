from __future__ import annotations
from typing import *
from ..core.environment import *

def main():
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
    
    for i in range(50*map.num_nodes**4):
        good_kitty.act()
        evil_kitty.act()
        good_kitty.forward_train()
    
    for index in range(64):
        i, j, k = good_kitty.qlearn_algorithm.decode_qtable_index(index)
        print(f'({i}, {j}, {k}): {good_kitty.qlearn_algorithm.qtable[index]}')
    
    return

main()