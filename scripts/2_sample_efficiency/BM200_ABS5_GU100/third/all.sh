#!/bin/bash

# run collect_s${i}.sh
for i in 1 2 3 4
do
    ./scripts/2_sample_efficiency/BM200_ABS5_GU100/third/collect_s${i}.sh &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

# run train.sh
./scripts/2_sample_efficiency/BM200_ABS5_GU100/third/train.sh

# run test.sh
# ./scripts/2_sample_efficiency/BM200_ABS5_GU100/third/test.sh

# run eval.sh
# ./scripts/2_sample_efficiency/BM200_ABS5_GU100/third/eval.sh
