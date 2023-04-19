#!/bin/bash

d=$(date +"%d%m%Y_%H%M%S")
res=$d"_dqn_result.txt"

for _ in {1..10}
do
  # shellcheck disable=SC2046
  python dqn.py>> $res
done