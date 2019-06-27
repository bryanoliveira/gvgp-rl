from algorithms._interface import RLInterface
import torch.nn as nn


import copy
from multiprocessing.pool import ThreadPool
import pickle
import time

import numpy as np
import torch


class EvolutionModule:

    def __init__(
        self, 
        weights, 
        reward_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None
    ):
        np.random.seed(int(time.time()))
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        self.decay = decay
        self.sigma_decay = sigma_decay
        self.pool = ThreadPool(threadcount)
        self.pool.daemon = True
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path


    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if no_jitter:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.SIGMA * population[i]).float()
                if self.cuda:
                    jittered = jittered.cuda()
                new_weights.append(param.data + jittered)
        return new_weights


    def run(self, iterations, print_step=10):
        print("Evolving %d generations." % iterations)
        for iteration in range(iterations):
            print('Generation: %d' % iteration)
            population = []
            for _ in range(self.POPULATION_SIZE):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                population.append(x)

            rewards = self.pool.map(
                self.reward_function, 
                [self.jitter_weights(copy.deepcopy(self.weights), population=pop) for pop in population]
            )
            if np.std(rewards) != 0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
                    if self.cuda:
                        rewards_pop = rewards_pop.cuda()
                    param.data = param.data + self.LEARNING_RATE / (self.POPULATION_SIZE * self.SIGMA) * rewards_pop

                    self.LEARNING_RATE *= self.decay
                    self.SIGMA *= self.sigma_decay

            if (iteration+1) % print_step == 0:
                test_reward = self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights), no_jitter=True), render=self.render_test
                )
                print('Generation: %d | Reward: %f' % (iteration+1, test_reward))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))
                
                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                        return self.weights

        return self.weights
        

class ES(RLInterface):
    def __init__(
        self,
        env_factory, 
        save_load_path = "trained_models", 
        skip_load = False,
        render = False,
        n_workers = mp.cpu_count(), 
        ):
        
        super(ES, self).__init__()

        self.name = "ES"

    def get_reward(weights, model, render=False):
        global env
        global max_episode_length

        cloned_model = copy.deepcopy(model)
        for i, param in enumerate(cloned_model.parameters()):
            try:
                param.data = weights[i]
            except:
                param.data = weights[i].data

        obs = env.reset()
        done = False
        total_reward = 0
        i = 0
        while not done and i < max_episode_length:
            if render:
                env.render()
            with torch.no_grad():
                prediction = cloned_model(obs.unsqueeze(0))
                action = prediction.data.max(1)[1].item()
            obs, reward, done = env.step(action)

            total_reward += reward 
            i += 1
        env.close()
        return total_reward


    partial_func = partial(get_reward, model=model)
    mother_parameters = list(model.parameters())

    es = EvolutionModule(
        mother_parameters, partial_func, population_size=args.population,
        sigma=0.01, learning_rate=args.lr, decay=0.9999,
        reward_goal=600, consecutive_goal_stopping=10, threadcount=args.threads,
        cuda=(args.device == torch.device('cuda')), render_test=args.render, save_path=os.path.abspath(args.weights_path)
    )

    start = time.time()
    final_weights = es.run(args.generations, print_step=args.print_steps)
    end = time.time() - start

    pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))
    print("Saved final weights to " + args.weights_path)


    reward = partial_func(final_weights, render=True)

    print("Reward from final weights: " + str(reward))
    print("Time to completion: " + str(end))
