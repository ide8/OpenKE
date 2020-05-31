#!/bin/bash

i=0
for exp in TransD TransE TransE_adv_sl TransH SimplE DistMult DistMult_adv HolE ComplEx RotatE_adv Analogy RESCAL
do
  for benchmark in FB13 FB15K FB15K237 NELL-995 WN11 WN18 WN18RR YAGO3-10
  do
    python train.py --exp $exp --benchmark $benchmark --n_epochs 1000 &
    sleep 3
    if (( $i % 2 )); then
      sleep 1400
    fi
    let i+=1
  done
done
