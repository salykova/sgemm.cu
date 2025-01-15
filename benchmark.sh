#!/bin/bash

npts=93
matsize_step=128
matsize_min=1024
warmup_matsize=4096
warmup_iter=500
CUBLAS=0
lock_clocks=0

bash ./scripts/enable_pmode.sh > /dev/null
if [ $lock_clocks -eq 1 ]
then
    bash ./scripts/lock_clocks.sh > /dev/null
else
    bash ./scripts/reset_clocks.sh > /dev/null
fi

bench_save_dir="benchmark_results"
mkdir -p ${bench_save_dir}
file_name="sgemm.cu"

sudo rm -f -r build
cmake -B build -S . -DGPUCC=$1 -DCUBLAS=0
cmake --build build --target benchmark
./build/benchmark --fname=${file_name} --savedir=${bench_save_dir} --wniter=${warmup_iter} \
                     --wmsize=${warmup_matsize} --mmin=${matsize_min} --mstep=${matsize_step} \
                     --npts=${npts}

if [ $CUBLAS -eq 1 ]
then
cmake -B build -S . -DGPUCC=$1 -DCUBLAS=1
cmake --build build --target benchmark
./build/benchmark --fname=${file_name} --savedir=${bench_save_dir} --wniter=${warmup_iter} \
                     --wmsize=${warmup_matsize} --mmin=${matsize_min} --mstep=${matsize_step} \
                     --npts=${npts}
fi

if [ $lock_clocks -eq 1 ]
then
    bash ./scripts/reset_clocks.sh > /dev/null
fi
bash ./scripts/disable_pmode.sh > /dev/null