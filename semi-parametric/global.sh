#!/bin/bash
END=14
for i in $(seq 0 $END)
do
	declare -a ind=$i
	grep -rl ind run.sh | xargs sed 's/ind/'$ind'/g' | sbatch
done