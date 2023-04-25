#!/bin/bash

d=$(date +"%d%m%Y_%H%M%S")
res=$d"_qlearning_result.txt"

for _ in {1..10}
do
  # shellcheck disable=SC2046
  python qlearning.py>> $res
done