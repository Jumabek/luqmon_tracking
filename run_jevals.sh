#!/bin/bash

# List of tracking methods
tracking_methods=("ocsort" "deepocsort" "bytetrack" "strongsort" "botsort")
#tracking_methods=("strongsort")

# # Loop through each tracking method and run the command
for tracking_method in "${tracking_methods[@]}"; do
    python examples/val.py --yolo-model yoloX_m --tracking-method "$tracking_method" --benchmark MOT17
done