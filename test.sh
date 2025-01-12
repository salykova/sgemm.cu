#!/bin/bash

matsize_max=3072
matsize_min=2
matsize_step=1

save_dir="test_results"
mkdir -p ${save_dir}

sudo rm -r -f build
cmake -B build -S . -DGPUCC=$1
cmake --build build --target test
./build/test --savedir=${test_save_dir} --mmax=${matsize_max} --mmin=${matsize_min} --mstep=${matsize_step}