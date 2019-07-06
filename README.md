# General Video Game Playing Reinforcement Learning Agents
A collection of reinforcement learning algorithms applied to General Video Game Playing. A good overview can be found [on this paper](https://arxiv.org/pdf/1802.10363.pdf).

## Requirements
- Python3
- Numpy
- Torch
- Gym
- Atari-py
- TensorboardX
- OpenCV 2

## Training
To quickly start training, run: 
- `python3 main.py --game GAME_NAME --wrapper WRAPPER --model MODEL`

Or run `python3 main.py --help` to see all available options.

Example:
- `python3 main.py --game SpaceInvadersNoFrameskip-v0 --wrapper atari_conv --model a3c_conv`

To use Atari's image observation with `atari_conv` wrapper, GAME_NAME must contain `NoFrameskip` in the name.

## Testing
To test, you may use `--play --render` options:
- `python3 main.py --game GAME_NAME --wrapper WRAPPER --model MODEL --play --render`

Example:
- `python3 main.py --game SpaceInvadersNoFrameskip-v0 --wrapper atari_conv --model a3c_conv --play --render`

You can specify `--random` to run a random agent with the same configs and collect statistics. The `--render` option can be also specified on training to see Worker nº 0's performance.

## References
 - [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
 - [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
 - [Evolution Strategies](https://arxiv.org/pdf/1703.03864.pdf)
