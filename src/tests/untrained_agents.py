from __future__ import annotations
from typing import *
from ..core.environment import *

def main():
    map = Map({
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 4, 5],
        3: [0, 4],
        4: [1, 2, 3, 5],
        5: [2, 4]
    })
    evil_kitty = Kitty(node=1, kitty_type=Kitty.types.EVIL_KITTY)
    good_kitty = Kitty(node=5, kitty_type=Kitty.types.GOOD_KITTY)
    goal_node = 3
    initial_state = State(evil_kitty=evil_kitty, good_kitty=good_kitty, goal_node=goal_node, map=map)
    environment = Environment(initial_state)
    environment.logs = True
    environment.run()
    return

main()