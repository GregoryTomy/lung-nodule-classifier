#!/bin/bash 


### increase/decrease num-workers as per your available CPU cores

python -m src.prepcache --num-workers 8 --batch-size 600