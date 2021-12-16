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
                              find_KMEANS_P_ABS_callback=scenario.find_KMEANS_P_ABS,
                              get_P_GU_with_augmentation_callback=scenario.get_P_GU_with_augmentation,
                              get_P_ABS_with_augmentation_callback=scenario.get_P_ABS_with_augmentation,
                              get_P_CGU_with_augmentation_callback=scenario.get_P_CGU_with_augmentation,

                              get_entities_statuses_callback=scenario.get_entities_statuses)
    elif args.scenario == "precise":
        env = SiteSpecificEnv(args=args,
                              world=world,
                              reset_world_callback=scenario.reset_world,
                              get_states_callback=scenario.get_states,
                              get_rewards_callback=scenario.get_rewards,
                              get_costs_callback=scenario.get_costs,
                              sample_actions_callback=scenario.sample_actions,
                              get_action_filters_callback=scenario.get_action_filters,
                              
                              get_entities_statuses_callback=scenario.get_entities_statuses)
    return env