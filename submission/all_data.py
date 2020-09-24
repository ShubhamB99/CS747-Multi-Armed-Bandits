import numpy as np

file_path = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algorithm = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
horizon = [100, 400, 1600, 6400, 25600, 102400]
for file in file_path:
  for algo in algorithm:
    for hor in horizon:
      for seed in np.arange(0,50):
        print("python3 bandit.py --instance {i} --algorithm {j} --randomSeed {k} --epsilon 0.02 --horizon {t}".format(i=file, j=algo, k=seed, t=hor))