# LunarLander v2
## The description of the task
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

Source: https://gym.openai.com/envs/LunarLander-v2/

## Approach to resolve the task

To resolve the task has been used a deep Q-network (DQN), implemented by Keras framework. The best model and optimal hyper parameters found by a grid search, running `lunarlander.py` script.

![Ideal landing. 2010th episode](lunarlander.gif)

[More info.](https://github.com/frizner/LunarLander-v2/blob/master/README.ipynb)
