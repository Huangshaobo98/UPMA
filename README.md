# Age of Information (AoI) optimization by joint UAV and worker nodes based on deep reinforcement learning algorithm D3QN
## Finished progress
1. Network model
2. D3QN adaptation
3. Energy model adaptation
4. Basic bug fix
5. AoI model adaptation
6. Data persistence
7. Train
8. Trust model (direct trust and recommendation trust)
9. Motion model of worker node (Not provide)
10. Comparison with traditional strategies

## Current work
1. Energy consumption of data transmit
2. More network scales ?



## How to run
```shell
# Train by default
python3 ./main.py 
# Train by adding parameters
python3 ./main.py -train
# Continuing training
python3 ./main.py -train -continue
# Test on
python3 ./main.py -test 
# Train/test data analysis (unfinished)
python3 ./main.py -analysis -train
# Console log on
python3 ./main.py -console
# File log on (save at ./save/log by default)
python3 ./main.py -file_log

# Other parameters
python3 ./main.py -lr 0.001 # learning rate
python3 ./main.py -batch 256 # batch size
python3 ./main.py -gamma 0.75 # reward discount rate
python3 ./main.py -decay 0.99995 # epsilon decay
python3 ./main.py -sensor 1000 # sensor number
python3 ./main.py -worker 2 # worker number