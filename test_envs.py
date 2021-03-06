# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import numpy as np
from pprint import pprint
from config import get_config
from common import make_env


def test_pattern():
    env = make_env(args, "train")
    env.reset()
    env.render()

    pprint(env.get_all_Ps())

    env.walk()
    env.render()

    kmeans_P_ABS = env.find_KMEANS_P_ABS()
    env.step(kmeans_P_ABS)
    env.render()
    pprint(env.get_all_Ps())

def test_precise():
    env = make_env(args, "train")
    env.reset()
    env.render()

    pprint(env.get_states())

    env.walk()
    env.render()
    while True:
        # actions = env.sample_actions()
        actions = np.ones((env.world.n_ABS,), dtype=np.int32)
        pprint(actions)
        env.step(actions)
        env.render()

        print(f"states: {env.get_states()}")
        print(f"filters: {env.get_action_filters()}")

if __name__ == "__main__":
    
    parser = get_config()
    args = parser.parse_args()
    pprint(args)

    if args.scenario == "pattern":
        test_pattern()
    elif args.scenario == "precise":
        test_precise()