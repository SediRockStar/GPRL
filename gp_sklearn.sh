#!/bin/bash

kernel='Matern'
T=1

d=$(date +"%d%m%Y_%H%M%S")
resDir="Data/Experiment_"
resDir+=$d
resDir+=_$kernel
resDir+=$T
[ -d Data ] || mkdir Data # if Data folder does not exist, create it
# shellcheck disable=SC2046
[ -d $resDir ] || mkdir $resDir # if result_dir does not exist, create it

res=$resDir"/result.txt"


touch $res

for i in {1..10}
do
  # shellcheck disable=SC2046
  python GPRL_sklearn.py -T $T -g 0.8 -ker $kernel -p True -dir $resDir>> $res
done