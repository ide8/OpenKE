#!/bin/bash

for exp in TransE TransH TransD DistMult
do
  for benchmark in FB15K237 WN18
  do
    python train.py --exp $exp --benchmark $benchmark --n_epochs 2000 &
    sleep 3
  done
  sleep 2500
done
