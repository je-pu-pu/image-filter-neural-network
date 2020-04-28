#!/bin/bash
d=`date +%Y%m%d-%H%M%S`
ffmpeg -r 60 -i ./output/0/0-%d.png -vb 5M  ./output/$d-0.mp4
ffmpeg -r 60 -i ./output/1/1-%d.png -vb 5M  ./output/$d-1.mp4
