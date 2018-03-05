#!/bin/bash
# this script tokenizes a certain field in the input field using CMU Tweet NLP

# Arguments:
#  $1 : input file
#  $2 : field to tokenize
#  $3 : output file

paste <(. twokenize.sh <(cut -f $2 $1)) $1 | \
    awk -v f="$(($2+2))" 'BEGIN { FS="\t" }{ $f=$1; for (i = 3; i < NF; i++) { printf("%s\t", $i); } printf("%s\n", $NF) }' > $3

