#!/bin/bash

./do_gap_pagerank $1 > res.txt

grep -i "SPMV perf" res.txt > perf.txt
grep -i "SPMV time" res.txt > time.txt
grep -i "SPMV BW" res.txt > bw.txt

filename="perf.txt"
n=1
sum=0.0
while read line; do
    n=$((n+1))
    NUMBER=$(echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    sum=$(echo "$sum + $NUMBER" | bc -l)
done < $filename
div=$n.0
avg_perf=$(echo "$sum / ($n.0)" | bc -l)

filename="time.txt"
n=1
sum=0.0
while read line; do
    n=$((n+1))
    NUMBER=$(echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    sum=$(echo "$sum + $NUMBER" | bc -l)
done < $filename
div=$n.0
avg_time=$(echo "$sum / ($n.0)" | bc -l)

filename="bw.txt"
n=1
sum=0.0
while read line; do
    n=$((n+1))
    NUMBER=$(echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    sum=$(echo "$sum + $NUMBER" | bc -l)
done < $filename
div=$n.0
avg_bw=$(echo "$sum / ($n.0)" | bc -l)

echo "avg time: "$avg_time
echo "avg perf: "$avg_perf
echo "avg bw: "$avg_bw
