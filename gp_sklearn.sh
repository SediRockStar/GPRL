#!/bin/bash


kernel=("Matern", "RBF", "RetionalQuadratic", "ExpSineSquared")
T=(10, 25, 50)
[ -d Data ] || mkdir Data

for (( i=0; i<${#kernel[@]}; i++ ))
do
  for (( j=0; j<${#T[@]}; j++ ))
  do
    d=$(date +"%d%m%Y_%H%M%S")
    resDir="Data/Experiment_"
    resDir+=$d
    resDir+=_${kernel[i]}
    resDir+=_${T[j]}
     # if Data folder does not exist, create it
    [ -d $resDir ] || mkdir $resDir # if result_dir does not exist, create it
    res=$resDir"/result.txt"
    touch $res
    for (( k=1; k<=10; k++ ))
    do
      python GPRL_sklearn.py -T ${T[j]} -g 0.9 -ker ${kernel[i]} -p True -dir $resDir >> $res
    done
  done
done
