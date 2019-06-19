import argparse

import algorithms.a3c as a3c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithms')
    parser.add_argument('--wrapper', type=str, default='gvgai', help='Game emulator wrapper framework')  # gym/gvgai
    parser.add_argument('--game', type=str, default='gvgai-cec1-lvl0-v0', help='ATARI game')  # default='space_invaders' gvgai-cec1-lvl0-v0
    parser.add_argument('--model', type=str, default='trained_models', help='Pretrained model')
    parser.add_argument('--skip-load', action='store_true', help='Skip loading the pretrained model')

    # Setup
    args = parser.parse_args()

    if args.wrapper == 'atari':
        from envs.atari import Env
        if args.game == 'gvgai-cec1-lvl0-v0':
            args.game = 'space_invaders'
    elif args.wrapper == 'gvgai' or args.game == "gvgai-combo":
        from envs.gvgai import Env
    elif args.wrapper == 'gym':
        from envs.gym import Env
        if args.game == 'gvgai-cec1-lvl0-v0':
            args.game = 'CartPole-v0'

    factory = False
    if args.game == "gvgai-combo":
        factory = [
            Env.factory("gvgai-cec1-lvl0-v0"), Env.factory("gvgai-cec1-lvl1-v0"), 
            Env.factory("gvgai-cec2-lvl0-v0"), Env.factory("gvgai-cec2-lvl1-v0"),
            Env.factory("gvgai-cec3-lvl0-v0"), Env.factory("gvgai-cec3-lvl1-v0")
        ]

    a3c.run(factory or Env.factory(args.game), load_path=args.model, skip_load=args.skip_load)