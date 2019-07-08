import argparse
import multiprocessing as mp
import logging
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithms')
    parser.add_argument('--wrapper', type=str, default='atari_conv', help='Game emulator wrapper framework')  # gym/gvgai
    parser.add_argument('--model', type=str, default='a3c_conv', help='RL model')
    parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v0', help='ATARI game')  # default='SpaceInvaders-v0' gvgai-cec1-lvl0-v0
    parser.add_argument('--save-load-path', type=str, default='trained_models', help='Pretrained model')
    parser.add_argument('--skip-load', action='store_true', help='Skip loading the pretrained model')
    parser.add_argument('--render', action='store_true', help='Render the 0th worker')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Number of worker processes')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma value')
    parser.add_argument('--update-global-delay', type=int, default=20, help='Delay to update global network')
    parser.add_argument('--max-eps', type=int, default=10000, help='Max number of episodes')
    parser.add_argument('--max-length', type=int, default=1000, help='Max length of each episode')
    parser.add_argument('--max-reward', type=int, default=1000, help='Reward considered to be a win')  # TODO
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--log', type=int, default=logging.INFO, help='Logging level')
    parser.add_argument('--play', action='store_true', help='Play game')
    parser.add_argument('--random', action='store_true', help='Play with random agent')
    parser.add_argument('--game-plays', type=int, default=5, help='Number of game plays')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Number of episode between each checkpoint')

    # Setup
    args = parser.parse_args()

    logging.basicConfig(
        filename='output.log', 
        filemode='w',
        level=args.log, 
        format='(%(levelname)s) %(asctime)s | \033[0;1m%(filename)s -> %(funcName)s\033[0m: \t %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if args.wrapper == 'gvgai' or args.game == "gvgai-combo":
        from envs.gvgai import Env
    elif args.wrapper == 'atari_conv':
        from envs.atari_conv import Env
    elif args.wrapper == 'atari':
        from envs.atari import Env
        if args.game == 'gvgai-cec1-lvl0-v0':
            args.game = 'SpaceInvaders-v0'
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

    if args.play:
        args.workers = 0  # it won't be a parallel worker

    if args.model == 'a3c':

        from algorithms.a3c import A3C

        a3c = A3C(
            env_factory = factory or Env.factory(args.game), 
            play = args.play,
            save_load_path = args.save_load_path,
            skip_load = args.skip_load,
            render = args.render,
            n_workers = args.workers,
            gamma = args.gamma,
            update_global_delay = args.update_global_delay,
            checkpoint_interval = args.checkpoint_interval,
            max_eps = args.max_eps,
            max_length = args.max_length
        )

        if args.play:
            a3c.play(args.game_plays)
        else:
            try:
                a3c.run()
            except Exception as e:
                logging.error(str(e))

    elif args.model == 'a3c_conv':

        from algorithms.a3c_conv import A3C

        a3c = A3C(
            env_factory = factory or Env.factory(args.game), 
            play = args.play,
            save_load_path = args.save_load_path,
            skip_load = args.skip_load,
            render = args.render,
            n_workers = args.workers,
            gamma = args.gamma,
            update_global_delay = args.update_global_delay,
            checkpoint_interval = args.checkpoint_interval,
            max_eps = args.max_eps,
            max_length = args.max_length,
            random = args.random
        )

        if args.play:
            a3c.play(args.game_plays)
        else:
            try:
                a3c.run()
            except Exception as e:
                logging.error(str(e))
    
    elif args.model == 'a2c':

        from algorithms.a2c_conv import A2C
        
        a2c = A2C(
            env_factory = factory or Env.factory(args.game),
            save_load_path = args.save_load_path,
            skip_load = args.skip_load,
            render = args.render,
            n_workers = args.workers,
            cuda = args.cuda,
            gamma = args.gamma,
            max_eps = args.max_eps,
            max_length = args.max_length
        )
        a2c.run()