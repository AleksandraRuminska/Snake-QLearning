#!/usr/bin/env python

from __future__ import division, print_function

import pandas as pd
import numpy as np
import random
import time
import sys
import gym
import matplotlib.pyplot as plt
from optparse import OptionParser
import csv

from gym_snake.envs.grid.base_grid import BaseGrid
from gym_snake.envs.constants import Action4
from PyQt5.QtCore import Qt

is_done = False
epsilon = 0


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym-snake environment to load",
        default='Snake-16x16-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    newtable = False
    if newtable:
        BaseGrid.combine_states(env.grid)
        Scores = open('scores.csv', 'w')
    else:
        Scores = open('scores.csv', 'a')
        open('scores1.csv', 'a')
        open('scores2.csv', 'a')
    writer = csv.writer(Scores)

    apples_eaten = 0

    # Learning rate
    alpha = 0.5

    # Discount rate
    gamma = 0.8

    # Explore rate
    # # Use if decay epsilon-greedy policy
    global epsilon

    # # If decay epsilon-greedy policy comment this line
    # epsilon = 0.1

    # Q-TABLE
    combinations = BaseGrid.read_table()

    def prepare_chart(scores):
        total = 0
        avg = []
        y = scores[0]
        length = len(y)
        for i in range(length):
            total = total + scores[0][i]
            avg.append(total / (i + 1))
        x = np.arange(length)
        return x, avg

    def plotSnakeScore():
        # Plot with decay epsilon-greedy policy
        sc_decay_epsilon_greedy = pd.read_csv('scores.csv', sep=',', header=None)

        x_decay_epsilon_greedy, avg_decay_epsilon_greedy = prepare_chart(sc_decay_epsilon_greedy)

        plt.plot(x_decay_epsilon_greedy, sc_decay_epsilon_greedy, label="Score")
        plt.plot(x_decay_epsilon_greedy, avg_decay_epsilon_greedy, label="Average score")
        plt.xlabel('Number of episodes')
        plt.ylabel('Score')
        plt.title('Average number of eaten apples vs number of episode')
        plt.legend()
        plt.show()

        # Compariosn plot between different policies of taking action
        sc_epsilon_greedy = pd.read_csv('scores2.csv', sep=',', header=None)
        sc_greedy = pd.read_csv('scores1.csv', sep=',', header=None)
        sc_decay_epsilon_greedy = pd.read_csv('scores.csv', sep=',', header=None)

        x_greedy, avg_greedy = prepare_chart(sc_greedy)
        x_epsilon_greedy, avg_epsilon_greedy = prepare_chart(sc_epsilon_greedy)
        x_decay_epsilon_greedy, avg_decay_epsilon_greedy = prepare_chart(sc_decay_epsilon_greedy)

        plt.plot(x_greedy, avg_greedy, label="Greedy")
        plt.plot(x_epsilon_greedy, avg_epsilon_greedy, label="Epsilon-greedy")
        plt.plot(x_decay_epsilon_greedy, avg_decay_epsilon_greedy, label="Decay epsilon-greedy")
        plt.xlabel('Number of episodes')
        plt.ylabel('Score')
        plt.title('Average number of eaten apples vs number of episode')
        plt.legend()

        plt.show()

    def keyDownCb(keyName):
        global is_done
        if keyName == Qt.Key_P:
            plotSnakeScore()
            sys.exit(0)

    def define_action(action_number):
        case = {
            0: Action4.forward,
            1: Action4.right,
            2: Action4.down,
            3: Action4.left
        }
        return case.get(action_number)

    def get_action(value):

        if random.uniform(0, 1) < epsilon:
            action_list = np.where(np.array(value) >= 0)[0]
            if not action_list.any():
                action_number = random.randint(0, 3)
            else:
                action_number = random.choice(action_list)
            searching_value = value[action_number]
        else:
            searching_value = max(value)
            action_list = np.where(np.array(value) == searching_value)[0]
            action_number = random.choice(action_list)

        action = define_action(action_number)

        return action, action_number, searching_value

    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    env.reset()
    done = False

    state = BaseGrid.define_state(env.grid)

    while not done:
        value = combinations.get(state)
        action, index, maxValue = get_action(value)
        obs, reward, done, info = env.step(action)
        if reward == 200:
            apples_eaten += 1
        env.render()
        time.sleep(0.03)
        newstate = BaseGrid.define_state(env.grid)
        if not done:
            newvalue = combinations.get(newstate)
            futureValue = max(newvalue)
            # bellman equation
            newActual = maxValue + alpha * (reward + gamma * futureValue - maxValue)
            value[index] = newActual
        if done:
            newActual = maxValue + alpha * (reward + gamma * 0 - maxValue)
            value[index] = newActual
            writer.writerow([apples_eaten])
        state = newstate

    BaseGrid.save_table(env.grid, combinations)
    Scores.close()


while True:
    # Use if decay epsilon-greedy policy
    epsilon = epsilon - 0.00007
    if epsilon <= 0:
        epsilon = 0
    if __name__ == "__main__":
        main()

# if __name__ == "__main__":
#     main()
