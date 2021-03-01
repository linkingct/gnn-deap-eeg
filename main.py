#!/usr/bin/env python

import argparse
from train import train
from test import test

parser = argparse.ArgumentParser(description='Process eeg emotions. http://www.eecs.qmul.ac.uk/mmv/datasets/deap/')
parser.add_argument('-t', '--n_targets', choices=[1,2,3,4], type=int, default=4,
                    help='Number of targets to be used. Default is all (4)')
parser.add_argument('-pf', '--participant_from', choices=range(1,32), type=int, default=1,
                    help='Which participant data to be used')
parser.add_argument('-pt', '--participant_to', choices=range(2,33), type=int, default=2,
                    help='Which participant data to be used')
parser.add_argument('-bs', '--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-me', '--max_epoch', type=int, default=1,
                    help='Max epochs for training')
parser.add_argument('-dst','--dont_shuffle_train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')

args = parser.parse_args()

if args.test:
    test(args)
else:
    train(args)



