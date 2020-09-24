import sys, argparse
from samplingAlgos import *

parser = argparse.ArgumentParser(description='Process inputs to multi-armed bandit algorithms')
parser.add_argument('--instance', help='Path to instance of file containing true mean of arms')
parser.add_argument('--algorithm', help='Choose one of [epsilon-greedy, ucb, kl-ucb, thompson-sampling, thompson-sampling-with-hint]')
parser.add_argument('--randomSeed', help=' Random Seed, non-negative integer to create repeatable results')
parser.add_argument('--epsilon', help='Exploration probability, a number between [0, 1]')
parser.add_argument('--horizon', help='Horizon of trials, non-negative number')

args = parser.parse_args()

file_path = str(args.instance)
algo = str(args.algorithm)
seed = int(args.randomSeed)
eps = float(args.epsilon)
horizon = int(args.horizon)

true_mean = []
with open(file_path) as fp:
    lines = fp.readlines()
    for line in lines:
        true_mean.append(float(line))
true_mean = np.array(true_mean)
# print(true_mean)

if algo == 'epsilon-greedy':
    answer = eps_greedy(true_mean, horizon, eps, seed)
elif algo == 'ucb':
    answer = UCB(true_mean, horizon, seed)
elif algo == 'kl-ucb':
    answer = KL_UCB(true_mean, horizon, seed)
elif algo == 'thompson-sampling' or algo == 'thompson-sampling-with-hint':
    answer = thompson_sampling(true_mean, horizon, seed)


output_string = str(file_path) + ', ' + str(algo) + ', ' + str(seed) + ', ' + str(eps) + ', ' + str(horizon) + ', ' + str(answer) + '\n'
print(output_string)



