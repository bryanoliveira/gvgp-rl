import argparse

#from envs.atari import Env
from envs.gvgai import Env
import algorithms.a3c as a3c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithms')
    parser.add_argument('--wrapper', type=str, default='ale', help='Game emulator wrapper framework')  # gym/gvgai
    parser.add_argument('--game', type=str, default='gvgai-cec1-lvl0-v0', help='ATARI game')  # default='space_invaders' gvgai-cec1-lvl0-v0

    # Setup
    args = parser.parse_args()

    a3c.run(Env.factory(args.game))