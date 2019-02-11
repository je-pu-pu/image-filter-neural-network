#!/bin/bash
d=`date +%Y%m%d-%H%m%S`
ffmpeg -r 60 -i ./output/0-%d.png -vb 5M  $d-0.mp4
ffmpeg -r 60 -i ./output/1-%d.png -vb 5M  $d-1.mp4
