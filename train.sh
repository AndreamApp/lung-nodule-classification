#!/bin/sh
python3 main.py -g 2 -p truncate_500_5808 -k 0.5
python3 main.py -g 2 -p truncate_500_5808 -k 0.7
python3 main.py -g 2 -p truncate_400_5808 -k 1.0
python3 main.py -g 2 -p truncate_450_5808 -k 1.0
python3 main.py -g 2 -p truncate_500_5808 -k 1.0
