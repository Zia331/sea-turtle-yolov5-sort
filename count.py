import os
import math
import argparse
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
numbers = re.compile(r'(\d+)')
#create a list of zeros
id_buckets = [0]*(100000000)

def run(
    thres,
    source=FILE
):
    path = str(source)
    os.chdir(path)
    for infile in glob.glob('*.txt'):
        file_path = f"{path}\{infile}"
        read_and_count(file_path)
    
    id_buckets.sort(reverse=True)
    
    if type(thres) == "<class 'list'>":
        print("Total amount: ", total_up(thres[0]))
    else:
        print("Total amount: ", total_up(thres))

# Sort the files then read them in ascending order
# Source: https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def read_and_count(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(',')
            #print(tokens[5][:-3])
            i = int(tokens[5][:-3])
            id_buckets[i]+=1

def total_up(thres):
    weighted_n_tracked_times = 0
    count = 0
    n_tracked_ids = 0
    tracked_times = 0
    ave_tracked_times = 0
    #count for n tracked ids & total tracked times
    for i in range(100000000):
        if id_buckets[i] > 0:
            #print(id_buckets[i])
            n_tracked_ids+=1
            tracked_times+=id_buckets[i]
        else:
            break
    
    #count for average
    ave_tracked_times = int(tracked_times/n_tracked_ids)
    print("n: ", n_tracked_ids, "\ntracked times: ", tracked_times, "\nave: ", ave_tracked_times)

    #count standard deviation
    S = 0
    for i in range(n_tracked_ids):
        S += pow((id_buckets[i]-ave_tracked_times), 2)/(n_tracked_ids-1)
    S = math.sqrt(S)
    print("S: ", S)   
    
    #set the threshold
    if thres == 0:
        if ave_tracked_times <= int(S): 
            thres = ave_tracked_times
        else:
            #lower the thres 
            n = 0
            for i in range(n_tracked_ids):
                if id_buckets[i] < ave_tracked_times:
                    break
                n += 1
            for i in range(n):
                weighted_n_tracked_times += id_buckets[n_tracked_ids-i]
            thres = weighted_n_tracked_times/n

    
    for i in range(n_tracked_ids):
        if id_buckets[i] > thres:
            count+=1    
   
    plt.scatter(np.arange(0,n_tracked_ids), id_buckets[:n_tracked_ids], s=10)
    plt.xlim(0,n_tracked_ids)
    plt.ylim(0,id_buckets[0]+10)
    plt.xlabel('Sorted IDs')
    plt.ylabel('Times of Tracked IDs Detected')
    plt.show()
    
    return count


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thres",
        nargs="+",
        type=int,
        default = 0,
        help="counting threshold",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT,
        help="file path",
    )
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)