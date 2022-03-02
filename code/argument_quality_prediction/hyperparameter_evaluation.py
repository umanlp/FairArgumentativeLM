import glob
import numpy as np
import codecs
import csv
from scipy.stats import pearsonr

# get dev results
files = glob.glob("my_code/GAQ/gpt2/gpt2_od_qa_fusion/*/eval_results.txt")
scores = []
for file in files:
    with open(file) as f:
        for line in f.readlines():
            if "overall" in line:
                scores.append(float(line.split("\'pearsonr\': ")[1].split(", \'pearsonp\'")[0]))
max_index = np.argmax(scores)
max_config = files[max_index]
print("Best dev result: " + max_config)