# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
from pprint import pprint

from config import get_config
from common import make_env
def test(args, device=torch.device("cpu")):
    """
    To test emulator accuracy in site-specific env.
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    method_dir = os.path.join(run_dir, args.method)
    K = args.K
    
    # dirs
    ckpt_dir = os.path.join(method_dir, "emulator_ckpts")

    eval_env = make_env(args, 'eval')

    """Load emulator"""
    from algorithms.emulator import Emulator
    emulator = Emulator(args, device)
    emulator_state_dict = torch.load(os.path.join(ckpt_dir, "best_emulator.pt"))
    emulator.model.load_state_dict(emulator_state_dict)

    cnter = 0
    with torch.no_grad():
        while cnter < 10:
            cnter += 1

            eval_env.reset()
            eval_env.render()
            
            P_GU, P_ABS, P_CGU = eval_env.get_all_Ps()

            P_GUs = torch.FloatTensor(P_GU.reshape(1, 1, K, K)).to(device)
            P_ABSs = torch.FloatTensor(P_ABS.reshape(1, 1, K, K)).to(device)
            P_CGUs = torch.FloatTensor(P_CGU.reshape(1, 1, K, K)).to(device)

            P_rec_CGUs = emulator.model.predict(P_GUs, P_ABSs)

            pprint(f"{torch.sum(torch.abs(P_rec_CGUs - P_CGUs))}")

            CR = torch.sum(P_CGUs) / eval_env.world.n_ON_GU
            pCR = torch.sum(P_rec_CGUs) / eval_env.world.n_ON_GU

            pprint("[CGU] - [recons] == [diff]")
            pprint(f"{CR} - {pCR} == {CR - pCR}")

            pprint("")

if __name__ == "__main__":

    # get specs
    parser = get_config()
    args = parser.parse_args()
    pprint(args)

    # cuda
    torch.set_num_threads(1)
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("chosse to use cpu...")
        device = torch.device("cpu")

    test(args, device)