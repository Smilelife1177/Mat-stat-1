#!/bin/bash

n=9

for ((i=0; i<n; i++)); do
    line=""
    for ((j=0; j<n; j++)); do
        if (( i==0 || i==n-1 || j==0 || j==n-1 )); then
            line+="#"             
        elif (( j==n-i-1 )); then
            line+="*"             
        else
            line+=" "             
        fi
    done
    echo "$line"                 
done
