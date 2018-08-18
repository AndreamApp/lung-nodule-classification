#!/bin/sh
python3 main.py -g 2 -p truncate_100 -k 1.0 -t
python3 main.py -g 2 -p truncate_300 -k 1.0 -t
python3 main.py -g 2 -p truncate_350 -k 1.0 -t
python3 main.py -g 2 -p truncate_400 -k 1.0 -t
python3 main.py -g 2 -p truncate_450 -k 1.0 -t
python3 main.py -g 2 -p truncate_500 -k 1.0 -t
python3 main.py -g 2 -p truncate_550 -k 1.0 -t
python3 main.py -g 2 -p truncate_600 -k 1.0 -t

python3 main.py -g 2 -p truncate_400 -k 0.7 -t
python3 main.py -g 2 -p truncate_450 -k 0.7 -t
python3 main.py -g 2 -p truncate_500 -k 0.7 -t

python3 main.py -g 2 -p truncate_400 -k 0.5 -t
python3 main.py -g 2 -p truncate_450 -k 0.5 -t
python3 main.py -g 2 -p truncate_500 -k 0.5 -t
