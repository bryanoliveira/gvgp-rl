import argparse

import algorithms.a3c as a3c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithms')
    parser.add_argument('--wrapper', type=str, default='gvgai', help='Game emulator wrapper framework')  # gym/gvgai
    parser.add_argument('--game', type=str, default='gvgai-cec1-lvl0-v0', help='ATARI game')  # default='space_invaders' gvgai-cec1-lvl0-v0
    parser.add_argument('--model', type=str, default='results/model.pth', help='Pretrained model')

    # Setup
    args = parser.parse_args()

    if args.wrapper == 'atari':
        from envs.atari import Env
        if args.game == 'gvgai-cec1-lvl0-v0':
            args.game = 'space_invaders'
    elif args.wrapper == 'gvgai':
        from envs.gvgai import Env
    elif args.wrapper == 'gym':
        from envs.gym import Env
        if args.game == 'gvgai-cec1-lvl0-v0':
            args.game = 'SpaceInvaders-v0'

    a3c.run(Env.factory(args.game))