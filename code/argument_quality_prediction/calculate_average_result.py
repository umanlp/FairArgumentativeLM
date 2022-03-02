import glob
import numpy as np
import codecs
import os
import csv
from scipy.stats import pearsonr
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.colors

plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")

import random

matplotlib.rcParams.update({
    'font.size': 16
})


# get dev results
all_paths = glob.glob("my_code/IBMRank/bert_*seed*")
scores = {}
for path in all_paths:
    print(os.path.basename(path))
    test_name = os.path.basename(path)
    test_name = re.search('bert_(.*)_seed', test_name).group(1)
    result_files = glob.glob(path + "/*/test_results.txt")
    results = []
    for file in result_files:
        with open(file) as f:
            for line in f.readlines():
                if "overall" in line:
                    results.append(float(line.split("\'pearsonr\': ")[1].split(", \'pearsonp\'")[0]))
                    #results.append(float(line.split("\'spearmanr\': ")[1].split(", \'spearmanp\'")[0]))
    scores[test_name] = results



means = []
errors = []
df_values = []
for key in scores:
    mean_value = sum(scores[key]) / len(scores[key])
    sd = statistics.stdev(scores[key])
    print("Average Result for " + str(key) + " over " + str(len(scores[key])) + " seeds: " + str(mean_value) + " with std: " + str(sd))
    CI = stats.t.interval(0.95, len(scores[key]) - 1, loc=np.mean(scores[key]), scale=stats.sem(scores[key]))
    error = CI[1] - mean_value
    df_values.append([key,mean_value,error])


all_tests = scores.keys()
base_test = [string for string in all_tests if 'original' in string]
#base_test = ['od_deb']
all_tests = all_tests - base_test
base_test = base_test[0]
for test in all_tests:
    print("-----T-Test Evaluation-----")
    print('Evaluation of ' + str(base_test) + ' and ' + str(test))
    t_test_result = stats.ttest_ind(scores[base_test], scores[test])
    print(t_test_result)
