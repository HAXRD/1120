#!/bin/bash

# run collect_s${i}.sh
for i in 1 2 3 4
do
    ./scripts/1_self_comparison/BM100_ABS5_GU100/collect_s${i}.sh &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

# run train.sh
./scripts/1_self_comparison/BM100_ABS5_GU100/train.sh

# run test.sh
./scripts/1_self_comparison/BM100_ABS5_GU100/test.sh

# run eval.sh
./scripts/1_self_comparison/BM100_ABS5_GU100/eval.sh
