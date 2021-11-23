# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

from envs.sse.scenarios import load
from envs.sse.environment import SiteSpecificEnv

def SSEEnv(args, is_base, seed=0):

    # load scenario from script
    scenario = load(args.scenario + ".py").Scenario()

    # create world
    world = scenario.make_world(args, is_base, seed)

    # create env
    if args.scenario == "pattern":
        env = SiteSpecificEnv(args=args,
                              world=world,
                              reset_world_callback=scenario.reset_world,
                              get_P_GU_callback=scenario.get_P_GU,
                              get_P_ABS_callback=scenario.get_P_ABS,
                              get_P_CGU_callback=scenario.get_P_CGU,
                              find_KMEANS_P_ABS_callback=scenario.find_KMEANS_P_ABS)
    elif args.scenario == "precise":
        env = SiteSpecificEnv(args=args,
                              world=world,
                              reset_world_callback=scenario.reset_world,
                              get_GU_info_callback=scenario.get_GU,
                              get_ABS_info_callback=scenario.get_ABS,
                              get_CGU_info_callback=scenario.get_CGU,
                              find_KMEANS_ABS_info_callback=scenario.find_KMEANS_ABS_info)
    return env