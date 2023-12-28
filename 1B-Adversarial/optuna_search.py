import copy
import optuna
import subprocess
from multiAgents import MultiAgentSearchAgent
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

# Create a deep copy of ContestAgent
specified_cnt = 1

def objective(trial):
    
    # Define the hyperparameters to optimize
    FOOD_RECIPROCAL_COEFF = trial.suggest_int('FOOD_RECIPROCAL_COEFF', 0, 200)
    CAPSULE_RECIPROCAL_COEFF = trial.suggest_int('CAPSULE_RECIPROCAL_COEFF', 100, 900)
    RETURN_FOODS_COEFF = trial.suggest_int('RETURN_FOODS_COEFF', 0, 200)
    RETURN_CAPSULES_COEFF = trial.suggest_int('RETURN_CAPSULES_COEFF', 0, 10000)
    SCARED_REWARD_COEFF = trial.suggest_int('SCARED_REWARD_COEFF', -100, 400)
    GHOST_PUNISH_COEFF = trial.suggest_int('GHOST_PUNISH_COEFF', -5000, 0)
    # Set the new COEFF variables in ContestAgent
    with open(f'coeff{specified_cnt}.txt', 'w') as f:
        f.write(f"FOOD_RECIPROCAL_COEFF: {FOOD_RECIPROCAL_COEFF}\n")
        f.write(f"CAPSULE_RECIPROCAL_COEFF: {CAPSULE_RECIPROCAL_COEFF}\n")
        f.write(f"RETURN_FOODS_COEFF: {RETURN_FOODS_COEFF}\n")
        f.write(f"RETURN_CAPSULES_COEFF: {RETURN_CAPSULES_COEFF}\n")
        f.write(f"SCARED_REWARD_COEFF: {SCARED_REWARD_COEFF}\n")
        f.write(f"GHOST_PUNISH_COEFF: {GHOST_PUNISH_COEFF}\n")

    # Run NewContestAgent
    result = subprocess.run('python autograder.py -q q6 --no-graphics', stdout=subprocess.PIPE, shell=True, cwd='.')
    # Parse the result from the txt file
    with open('q6_objs.txt', 'r') as f:
        lines = f.readlines()
        avg_score = sum([float(line.strip()) for line in lines]) / len(lines)
    import time
    time.sleep(1)
    with open('q6_objs.txt', 'w') as _: 
        pass
    if int(lines[specified_cnt-1]) > 2900:
        raise optuna.exceptions.TrialPruned()
    # return avg_score
    return int(lines[specified_cnt-1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5000, show_progress_bar=True)
print("Best hyperparameters: ", study.best_params)
