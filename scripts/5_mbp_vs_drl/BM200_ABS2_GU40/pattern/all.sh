#!/bin/bash

# run collect_s${i}.sh
for i in 1 2 3 4
do
    ./scripts/5_mbp_vs_drl/BM200_ABS2_GU40/pattern/collect_s${i}.sh &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

# run train.sh
./scripts/5_mbp_vs_drl/BM200_ABS2_GU40/pattern/train.sh

# run test.sh
./scripts/5_mbp_vs_drl/BM200_ABS2_GU40/pattern/test.sh

# run eval.sh
./scripts/5_mbp_vs_drl/BM200_ABS2_GU40/pattern/eval.sh
