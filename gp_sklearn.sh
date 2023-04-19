#!/bin/bash

kernel='RBF'
d=$(date +"%d%m%Y_%H%M%S")
resDir="Data/Experiment_"
resDir+=$d
resDir+=_$kernel
[ -d Data ] || mkdir Data # if Data folder does not exist, create it
# shellcheck disable=SC2046
[ -d $resDir ] || mkdir $resDir # if result_dir does not exist, create it

res=$resDir"/result.txt"
touch $res


# shellcheck disable=SC2046
python GPRL_sklearn.py -T 8 -g 0.8 -ker $kernel -p True -dir $resDir>> $res