import random
import numpy as np

def eps_greedy(true_mean, horizon, eps, seed):
    random.seed(seed)
    num_arms = len(true_mean)
    emp_mean = np.zeros(num_arms, dtype = float)
    pull_count = np.zeros(num_arms, dtype = float)
    reward_count = np.zeros(num_arms, dtype = float)

    # Exploring num_arms times randomly to initialise
    for i in range(num_arms):
        arm_choice = random.randint(0, num_arms-1)
        pull_count[arm_choice] += 1 
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1

    emp_mean = np.divide(reward_count, pull_count, where=pull_count!=0)

    # Now exploring with probability eps and exploiting with (1-eps)
    for i in range(horizon-num_arms):
        pull_decision = random.random()

        if pull_decision <= eps :
            arm_choice = random.randint(0, num_arms-1)
            pull_count[arm_choice] += 1
            rand = random.random()
            if rand < true_mean[arm_choice]:
                reward_count[arm_choice] += 1

        elif pull_decision > eps:
            arm_choice = np.argmax(emp_mean)
            pull_count[arm_choice] += 1
            rand = random.random()
            if rand < true_mean[arm_choice]:
                reward_count[arm_choice] += 1

        emp_mean = np.divide(reward_count, pull_count, where=pull_count!=0)

    total_reward_count = np.sum(reward_count)
    max_reward_count = horizon * np.max(true_mean)
    regret = max_reward_count - total_reward_count

    return regret


def UCB(true_mean, horizon, seed):
    random.seed(seed)
    num_arms = len(true_mean)
    pull_count_t = np.zeros(num_arms, dtype = float)
    reward_count = np.zeros(num_arms, dtype = float)
    emp_mean = np.zeros(num_arms, dtype = float)
    ucb_score_t = np.zeros(num_arms, dtype = float)

    # Exploring all arms once in order, then num_arms times randomly to initialise
    for arm_choice in range(num_arms):
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1

    for i in range(num_arms, num_arms*2):
        arm_choice = random.randint(0, num_arms-1)
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1
    
    emp_mean = reward_count/pull_count_t

    # Now exploiting using UCB 
    for i in range(num_arms*2, horizon):
        ucb_score_t = emp_mean + np.sqrt(2*np.log(i)/pull_count_t)      # Check formula here
        arm_choice = np.argmax(ucb_score_t)
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1

        emp_mean[arm_choice] = reward_count[arm_choice]/pull_count_t[arm_choice]

    total_reward_count = np.sum(reward_count)
    max_reward_count = horizon * np.max(true_mean)
    regret = max_reward_count - total_reward_count

    return regret


def KL_UCB(true_mean, horizon, seed, precision = 1e-6):
    random.seed(seed)
    num_arms = len(true_mean)
    pull_count_t = np.zeros(num_arms, dtype = float)
    reward_count = np.zeros(num_arms, dtype = float)
    emp_mean = np.zeros(num_arms, dtype = float)
    kl_ucb_t = np.zeros(num_arms, dtype = float)

    # Calculating KL-Divergence
    def KL(a, b):
        thresh = 1e-15
        x = min(max(a, thresh), 1 - thresh)
        y = min(max(b, thresh), 1 - thresh)
        return x * np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

    # Calculating q for each arm
    def q_optimum(p_hat, t, uat, c = 0, precision = precision):
        kld_max = (np.log(t) + c*np.log(np.log(t)))/uat
        q_min = p_hat
        q_max = 1.0
        q_mid = (q_min + q_max)/2
        kld = KL(p_hat, q_mid)
        while kld > kld_max or (kld_max - kld) > precision:
            if kld > kld_max:
                q_max = q_mid 
            else:
                q_min = q_mid

            q_mid = (q_max + q_min)/2
            kld = KL(p_hat, q_mid)
            if (1-q_mid < precision):
                break
        return q_mid

    # Exploring every arm in order once to initialise
    for arm_choice in range(num_arms):
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1
    
    emp_mean = reward_count/pull_count_t

    for i in range(num_arms):
        kl_ucb_t[i] = q_optimum(emp_mean[i], num_arms, pull_count_t[i])

    # Now exploiting using KL-UCB
    for t in range(num_arms, horizon):
        arm_choice = np.argmax(kl_ucb_t)
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            reward_count[arm_choice] += 1

        emp_mean[arm_choice] = reward_count[arm_choice]/pull_count_t[arm_choice]

        for i in range(num_arms):
            kl_ucb_t[i] = q_optimum(emp_mean[i], t+1, pull_count_t[i])

    total_reward_count = np.sum(reward_count)
    max_reward_count = horizon * np.max(true_mean)
    regret = max_reward_count - total_reward_count

    return regret

    
def thompson_sampling(true_mean, horizon, seed):
    random.seed(seed) 
    num_arms = len(true_mean)
    success_t = np.zeros(num_arms)
    pull_count_t = np.zeros(num_arms)
    beta_t = np.zeros(num_arms)
    
    # Starting sampling by calculating belief of true mean of arms
    for t in range(horizon):
        for i in range(num_arms):
            beta_t[i] = random.betavariate((success_t[i] + 1), (pull_count_t[i] - success_t[i] + 1))
        arm_choice = np.argmax(beta_t)
        pull_count_t[arm_choice] += 1
        rand = random.random()
        if rand < true_mean[arm_choice]:
            success_t[arm_choice] += 1

    total_reward_count = np.sum(success_t)
    max_reward_count = horizon * np.max(true_mean)
    regret = max_reward_count - total_reward_count

    return regret