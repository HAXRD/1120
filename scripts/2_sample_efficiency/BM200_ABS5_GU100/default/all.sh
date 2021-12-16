#!/bin/bash

src=results_1_self_comparison/BM200_ABS5_GU100_default/
target=results_2_sample_efficiency/BM200_ABS5_GU100_default

# copy from results_1_self_comparison
cp -r ${src} ${target}
echo "copy from '${src}' to '${target}' done"
