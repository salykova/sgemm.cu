#!/bin/bash

base_clock=$(nvidia-smi base-clocks | grep -Eo '[0-9]{2,4}')
memory_clock=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Memory" | grep -om1 "[0-9]\+")
sudo nvidia-smi --lock-gpu-clocks=${base_clock}
sudo nvidia-smi --lock-memory-clocks=${memory_clock}