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
from chainer import variable
from chainer import links as L
from chainer import functions as F
from chainer import cuda

from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import misc

from chainerrl.optimizers import rmsprop_async
from chainerrl.optimizers import nonbias_weight_decay
from chainerrl.recurrent import state_kept
import net13 as net
import a3c09 as a3c

def phi(screens):
    assert len(screens) == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values

def train_single_process(make_agent, n_actions, outdir, make_env, steps, eval_n_runs,
        eval_frequency, xp, logger, max_episode_len=None, gamma=0.99, step_offset=0,
        eval_explorer=None):

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    evaluator = experiments.evaluator.AsyncEvaluator(
        n_runs=eval_n_runs, eval_frequency=eval_frequency, outdir=outdir,
        max_episode_len=max_episode_len, step_offset=step_offset,
        explorer=eval_explorer)

    misc.random_seed.set_random_seed(0)
    env = make_env(process_idx=0, test=False)
    eval_env = make_env(process_idx=0, test=True)
    agent = make_agent(process_idx=0, a2c=True)
    total_r = 0
    episode_r = 0
    local_t = 0
    state = env.reset()
    r = 0
    done = False
    base_lr = agent.optimizer.lr
    episode_len = 0
    action = 0
    action_array = xp.zeros((1, n_actions), dtype=np.float32)
    while True:
        total_r += r
        episode_r += r
        if done or episode_len == max_episode_len:
            agent.stop_episode_and_train(state, r, action_array, done)
            logger.info('local_step:%s lr:%s R:%s', local_t, agent.optimizer.lr, episode_r)
            logger.info('statistics:%s', agent.get_statistics())
            eval_score = evaluator.evaluate_if_necessary(local_t, env=eval_env, agent=agent)
            episode_r = 0
            state = env.reset()
            r = 0
            done = False
            episode_len = 0
        else:
            action = agent.act_and_train(state, r, action_array)
#            action = cuda.to_cpu(action)
            state, r, done, info = env.step(action)
            action_array.fill(0)
            action_array[0][action]=1
            # Get and increment the global counter
            local_t += 1
            episode_len += 1
            if local_t > steps:
                break
            agent.optimizer.lr = (steps - local_t - 1) / steps * base_lr
    if local_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)


def train_each_process(process_idx, make_env, make_agent, n_actions, steps, outdir, counter,
        max_episode_len, evaluator, shared_objects, logger):
    os.environ['OMP_NUM_THREADS'] = '1'
    misc.random_seed.set_random_seed(process_idx)
    env = make_env(process_idx, test=False)
    eval_env = make_env(process_idx, test=True)
    local_agent = make_agent(process_idx, a2c=False)
    for attr, shared in shared_objects.items():
        new_value = misc.async.synchronize_to_shared_objects(getattr(local_agent, attr), shared)
        setattr(local_agent, attr, new_value)
    local_agent.process_idx = process_idx
    total_r = 0
    episode_r = 0
    global_t = 0
    local_t = 0
    state = env.reset()
    r = 0
    done = False
    base_lr = local_agent.optimizer.lr
    episode_len = 0
    action = 0
    action_array = np.zeros((1, n_actions), dtype=np.float32)
    while True:
        total_r += r
        episode_r += r
        if done or episode_len == max_episode_len:
            local_agent.stop_episode_and_train(state, r, action_array, done)
            if process_idx == 0:
                logger.info('global_step:%s local_step:%s lr:%s R:%s', global_t, local_t, local_agent.optimizer.lr, episode_r)
                logger.info('statistics:%s', local_agent.get_statistics())
            eval_score = evaluator.evaluate_if_necessary(global_t, env=eval_env, agent=local_agent)
            episode_r = 0
            state = env.reset()
            r = 0
            done = False
            episode_len = 0
        else:
            action = local_agent.act_and_train(state, r, action_array)
            state, r, done, info = env.step(action)
            action_array.fill(0)
            action_array[0][action]=1
            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1
            episode_len += 1
            if global_t > steps:
                break
            local_agent.optimizer.lr = (steps - global_t - 1) / steps * base_lr
    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        local_agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)


def train_processes_async(outdir, n_process, n_actions, make_agent, make_env,
                steps, eval_frequency, eval_n_runs, max_episode_len, logger,
                gamma=0.99, step_offset=0, eval_explorer=None):

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'
    counter = mp.Value('l', 0)
    agent=make_agent(0, a2c=False)
    shared_objects = dict((attr, misc.async.as_shared_objects(getattr(agent, attr))) for attr in agent.shared_attributes)
    for attr, shared in shared_objects.items():
        new_value = misc.async.synchronize_to_shared_objects(getattr(agent, attr), shared)
        setattr(agent, attr, new_value)

    evaluator = experiments.evaluator.AsyncEvaluator(
        n_runs=eval_n_runs, eval_frequency=eval_frequency, outdir=outdir,
        max_episode_len=max_episode_len, step_offset=step_offset,
        explorer=eval_explorer)

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=train_each_process,
            args=(process_idx, make_env, make_agent, n_actions, steps, outdir,
                counter, max_episode_len, evaluator, shared_objects, logger)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


def main():
    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_process', type=int, default=1)
    parser.add_argument('--rom', type=str, default='pong')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--input_action', action='store_false', default=True)
    parser.add_argument('--skip_P', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default='./data')
    parser.add_argument('--use-sdl', action='store_false', default=True)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
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
    logging.basicConfig(level=logging.DEBUG)
    logger=logging.getLogger('mainlog')
    filehandler = logging.FileHandler(outdir + '/mainlog.log')
    filehandler.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)

    n_actions = ale.ALE(args.rom).number_of_actions
    model = net.SiS(n_actions, args.input_action, args.skip_P)
    opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
    opt.setup(model)

    if args.gpu >= 0:
#        cuda.check_cuda_available()
#        cuda.get_device(args.gpu).use()
#        model.to_gpu(device=args.gpu)
        xp = cuda.cupy
    else:
#        model.to_cpu()
        xp=np

    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(nonbias_weight_decay.NonbiasWeightDecay(args.weight_decay))


    def make_env(process_idx, test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl, treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
        return env

    def make_agent(process_idx, a2c):
        agent = a3c.A3C(model, opt, a2c=a2c, gpu=args.gpu, t_max=args.t_max, gamma=0.99, beta=args.beta, phi=phi, logger=logger)
        agent.process_idx=process_idx
        return agent

    if args.n_process==1:
        train_single_process(
            make_agent=make_agent,
            n_actions=n_actions,
            outdir=args.outdir,
            make_env=make_env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            max_episode_len=args.max_episode_len,
            xp=xp,
            logger=logger)
    elif args.gpu < 0:
        train_processes_async(
            make_agent=make_agent,
            n_actions=n_actions,
            outdir=args.outdir,
            n_process=args.n_process,
            make_env=make_env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            max_episode_len=args.max_episode_len,
            logger=logger)
    else:
        print('GPU not supported for asynchronous multiprocessing.')

if __name__ == '__main__':
    main()
