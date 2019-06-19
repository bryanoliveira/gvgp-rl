import argparse
import multiprocessing as mp

from algorithms.a3c import A3C

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithms')
    parser.add_argument('--wrapper', type=str, default='gvgai', help='Game emulator wrapper framework')  # gym/gvgai
    parser.add_argument('--game', type=str, default='gvgai-cec1-lvl0-v0', help='ATARI game')  # default='space_invaders' gvgai-cec1-lvl0-v0
    parser.add_argument('--save-load-path', type=str, default='trained_models', help='Pretrained model')
    parser.add_argument('--skip-load', action='store_true', help='Skip loading the pretrained model')
    parser.add_argument('--render', action='store_true', help='Render the 0th worker')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Number of worker processes')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma value')
    parser.add_argument('--update-global-delay', type=int, default=20, help='Delay to update global network')
    parser.add_argument('--max-eps', type=int, default=10000, help='Max number of episodes')
    parser.add_argument('--max-eps-length', type=int, default=1000, help='Max length of each episode')

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

    a3c = A3C(
        env_factory = factory or Env.factory(args.game), 
        save_load_path = args.save_load_path,
        skip_load = args.skip_load,
        render = args.render,
        n_workers = args.workers,
        gamma = args.gamma,
        update_global_delay = args.update_global_delay,
        max_eps = args.max_eps,
        max_eps_length = args.max_eps_length
    )
    a3c.run()