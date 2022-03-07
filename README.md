## Snake Q-learning

Project for AI classes at University written in Python adding the Q-learning algorithm for the existing Snake game. The Q-function that was implemented takes two input values: state and action. As the output it returns the expected future reward of that action at that state:

NewQ(st, at) = Q(st, at) + [R(st, at) + maxQ(st+1, a)-Q(st, at)]

**Steps of Q-learning algorithms**
- Build initial Q-table (where columns represent actions, rows represent states).
- Observe the current state.
- Choose an action.
- Perform an action.
- Measure reward.
- Update the Q-function. Go back to step 2.

**Rewarding system outline:**

A good move is the one when the snake eats the apple (only one instance of apple at a given time on the board), bad one when he dies. Any other move at this stage can be considered as ‘neutral’ for our game. 
- eaten apple: positive reward -  +10 
- died (hitting the wall or hitting own body) : negative reward - -100
- else: neutral - 0

**Actions:**

Agent can choose one of four action in each step:
- move up
- move down
- move right
- move left

All directions are determined in relation to the human eyes/computer screen.

**States:**

States of environment depend on the three facts: if a wall is in one of the adjacent cells, placement of the food relative to the snake head and if the snake’s body is in any of four adjacent cells to the head (the solution doesn't not provide that our agent will avoid enclosing nor will it check diagonal cells). 

Wall: 
- No/Up/Down - 0/1/2 - there is no such cell which can have a wall both above and below itself 
- No/Left/Right - 0/1/2 - there is no such cell which can have a wall both on left and right side 

Apple: 
- No/Up/Down - 0/1/2 - an apple cannot be above and below snake in the same time
- No/Left/Right - 0/1/2 - an apple cannot be on left and right side in the same time 

Body (no exclusion of opposite directions):
- No/Up - 0/1
- No/Down - 0/1
- No/Left - 0/1
- No/Right - 0/1

This combination gives 1296 states overall.  

**Experimenting**

To achieve the best result, the three policies with different exploration and exploitation were tested:
- Greedy policy
- Epsilon-greedy policy
- Decay epsilon-greedy policy

**Source Code**

For purposes of the project we decided to use shared on Github OpenAI Gym environment.
BSD 3-Clause License. Copyright (c) 2019, Telmo Correa. All rights reserved.
