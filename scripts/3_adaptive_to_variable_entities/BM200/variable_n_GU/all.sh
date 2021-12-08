#!/bin/bash

# run collect_s${i}.sh
for i in 1 2 3 4
do
    ./scripts/3_adaptive_to_variable_entities/BM200/variable_n_GU/collect_s${i}.sh &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

# run train.sh
./scripts/3_adaptive_to_variable_entities/BM200/variable_n_GU/train.sh

# run test.sh
./scripts/3_adaptive_to_variable_entities/BM200/variable_n_GU/test.sh

