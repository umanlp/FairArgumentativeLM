import re
from utils.target_terms import *
import random
import nltk
import itertools
from itertools import *
from nltk import tokenize
import pandas as pd
from tqdm import *
import argparse
from datasets import load_dataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Input file directory")
    parser.add_argument("--output_test",
                        type=str,
                        required=True,
                        help="The output test file")
    parser.add_argument("--output_train",
                        type=str,
                        required=True,
                        help="The output train file")
    args = parser.parse_args()

    dataset = load_dataset("text", data_files=args.input_file, split='train', cache_dir='cda_cache')

    dataset = dataset.train_test_split(test_size=0.2, shuffle= False)

    print(len(dataset['train']['text']))
    print(len(dataset['test']['text']))


    # Write to file for reconstructability
    with open(args.output_train, 'w', encoding='utf8') as training_file:
        for arg in dataset['train']['text']:
            training_file.write(arg)
            training_file.write('\n')

    with open(args.output_test, 'w', encoding='utf8') as test_file:
        for arg in dataset['test']['text']:
            test_file.write(arg)
            test_file.write('\n')





if __name__ == "__main__":
    main()