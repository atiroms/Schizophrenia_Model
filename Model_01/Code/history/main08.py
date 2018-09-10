from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import logging
import multiprocessing as mp
import os
import numpy as np

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import cuda

#from chainerrl.agents import a3c
from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policy
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function
from chainerrl.experiments.evaluator import AsyncEvaluator
from chainerrl.misc import random_seed
from chainerrl.misc import async

import net07 as net
import a3c03 as a3c


from chainer import variable


class A3CPredNet(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    def __init__(self, n_actions):
        self.head = net.SiS(n_actions)
        self.pi = policy.FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state, action_array):
        out = self.head(state, action_array)
        return self.pi(out), self.v(out)

def phi(screens):

    assert len(screens) == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values


def train_loop(process_idx, env, agent, n_actions, steps, outdir, counter, training_done,
               max_episode_len=None, evaluator=None, eval_env=None,
               logger=None):

    logger = logger or logging.getLogger(__name__)

    total_r = 0
    episode_r = 0
    global_t = 0
    local_t = 0
    state = env.reset()
    r = 0
    done = False
    base_lr = agent.optimizer.lr
    episode_len = 0
    successful = False

    action = 0
    action_array = np.zeros((1, n_actions), dtype=np.float32)

    while True:

        total_r += r
        episode_r += r

        if done or episode_len == max_episode_len:
            agent.stop_episode_and_train(state, r, action_array, done)
            if process_idx == 0:
                logger.info('global_step:%s local_step:%s lr:%s R:%s', global_t, local_t, agent.optimizer.lr, episode_r)
                logger.info('statistics:%s', agent.get_statistics())
            eval_score = evaluator.evaluate_if_necessary(global_t, env=eval_env, agent=agent)
            episode_r = 0
            state = env.reset()
            r = 0
            done = False
            episode_len = 0
        else:
            action = agent.act_and_train(state, r, action_array)
            state, r, done, info = env.step(action)
            action_array.fill(0)
            action_array[0][action]=1

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1
            episode_len += 1

            if global_t > steps or training_done.value:
                break

            agent.optimizer.lr = (steps - global_t - 1) / steps * base_lr

    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)

def extract_shared_objects_from_agent(agent):
    return dict((attr, async.as_shared_objects(getattr(agent, attr))) for attr in agent.shared_attributes)


def set_shared_objects(agent, shared_objects):
    for attr, shared in shared_objects.items():
        new_value = async.synchronize_to_shared_objects(getattr(agent, attr), shared)
        setattr(agent, attr, new_value)

def train_async(outdir, n_process, n_actions, make_env,
                steps=8 * 10 ** 7, eval_frequency=10 ** 6,
                eval_n_runs=10, gamma=0.99, max_episode_len=None,
                step_offset=0, eval_explorer=None, agent=None,
                logger=None):

    logger = logger or logging.getLogger(__name__)

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    counter = mp.Value('l', 0)
    training_done = mp.Value('b', False)  # bool

    shared_objects = extract_shared_objects_from_agent(agent)
    set_shared_objects(agent, shared_objects)

    print(logger)

    evaluator = AsyncEvaluator(
        n_runs=eval_n_runs,
        eval_frequency=eval_frequency, outdir=outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        explorer=eval_explorer)

    def run_func(process_idx):
        random_seed.set_random_seed(process_idx)

        env = make_env(process_idx, test=False)
        eval_env = make_env(process_idx, test=True)

        local_agent = agent
        local_agent.process_idx = process_idx

        train_loop(
            process_idx=process_idx,
            counter=counter,
            agent=local_agent,
            n_actions=n_actions,
            env=env,
            steps=steps,
            outdir=outdir,
            max_episode_len=max_episode_len,
            evaluator=evaluator,
            training_done=training_done,
            eval_env=eval_env,
            logger=logger)

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=run_func, args=(process_idx, )))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_process', type=int, default=2)
    parser.add_argument('--rom', type=str, default='pong')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--outdir', type=str, default='./data')
    parser.add_argument('--use-sdl', action='store_false', default=True)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true', default=False)
#    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 6)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    outdir = experiments.prepare_output_dir(args, args.outdir)

    n_actions = ale.ALE(args.rom).number_of_actions
    model = A3CPredNet(n_actions)

    optimizer = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
        print('Running on a GPU')
    else:
        xp=np
        print('Running on a CPU')

    optimizer.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        optimizer.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = a3c.A3C(model, optimizer, gpu=args.gpu, t_max=args.t_max, gamma=0.99, beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)

    def make_env(process_idx, test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl, treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
        return env

    if args.demo:
        env = make_env(0, True)
        mean, median, stdev = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev'.format(
            args.eval_n_runs, mean, median, stdev))
    else:
        train_async(
            agent=agent,
            n_actions=n_actions,
            outdir=args.outdir,
            n_process=args.n_process,
            make_env=make_env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            max_episode_len=args.max_episode_len)


if __name__ == '__main__':
    main()
