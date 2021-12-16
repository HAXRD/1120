#!/bin/bash

for n in 0 50 100 150 200 250 300 350 400 450 500 550 600
do
    python gen_terrain.py --n_BM ${n}
done
