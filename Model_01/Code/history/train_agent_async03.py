from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import logging
import multiprocessing as mp
import numpy as np
import os

from chainerrl.experiments.evaluator import AsyncEvaluator
from chainerrl.misc import async
from chainerrl.misc import random_seed

from chainer import variable


def train_loop(process_idx, env, agent, steps, outdir, counter, training_done,
               max_episode_len=None, evaluator=None, eval_env=None,
               successful_score=None, logger=None):

    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    total_r = 0
    episode_r = 0
    global_t = 0
    local_t = 0
    obs = env.reset()
    r = 0
    done = False
    base_lr = agent.optimizer.lr
    episode_len = 0
    successful = False

    action = 0
    action_array = variable.Variable(np.zeros((1, env.number_of_actions), dtype=np.float32))

    while True:

        total_r += r
        episode_r += r

        if done or episode_len == max_episode_len:
            agent.stop_episode_and_train(obs, r, done)
            if process_idx == 0:
                logger.info(
                    'outdir:%s global_step:%s local_step:%s lr:%s R:%s',
                    outdir, global_t, local_t, agent.optimizer.lr,
                    episode_r)
                logger.info('statistics:%s', agent.get_statistics())
            if evaluator is not None:
                eval_score = evaluator.evaluate_if_necessary(
                    global_t, env=eval_env, agent=agent)
                if (eval_score is not None and successful_score is not None and eval_score >= successful_score):
                    with training_done.get_lock():
                        if not training_done.value:
                            training_done.value = True
                            successful = True
                    # Break immediately in order to avoid an additional
                    # call of agent.act_and_train
                    break
            episode_r = 0
            obs = env.reset()
            r = 0
            done = False
            episode_len = 0
        else:
            action = agent.act_and_train(obs, r, action_array)
            obs, r, done, info = env.step(action)

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1
            episode_len += 1

            if global_t > steps or training_done.value:
                break

            agent.optimizer.lr = (steps - global_t - 1) / steps * base_lr

#    except KeyboardInterrupt:
#        if process_idx == 0:
#            # Save the current model before being killed
#            dirname = os.path.join(outdir, '{}_except'.format(global_t))
#            agent.save(dirname)
#            logger.warning('Saved the current model to %s', dirname)
#        raise

    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, 'successful')
        agent.save(dirname)
        logger.info('Saved the successful agent to %s', dirname)


def extract_shared_objects_from_agent(agent):
    return dict((attr, async.as_shared_objects(getattr(agent, attr))) for attr in agent.shared_attributes)


def set_shared_objects(agent, shared_objects):
    for attr, shared in shared_objects.items():
        new_value = async.synchronize_to_shared_objects(getattr(agent, attr), shared)
        setattr(agent, attr, new_value)


def train_agent_async(outdir, processes, make_env,
                      profile=False, steps=8 * 10 ** 7, eval_frequency=10 ** 6,
                      eval_n_runs=10, gamma=0.99, max_episode_len=None,
                      step_offset=0, successful_score=None,
                      eval_explorer=None,
                      agent=None, make_agent=None,
                      logger=None):
    """Train agent asynchronously.

    One of agent and make_agent must be specified.

    Args:
      agent (Agent): Agent to train
      make_agent (callable): (process_idx) -> Agent
      processes (int): Number of processes.
      make_env (callable): (process_idx, test) -> env
      model_opt (callable): () -> (models, optimizers)
      profile (bool): Profile if set True
      steps (int): Number of global time steps for training
    """

    logger = logger or logging.getLogger(__name__)

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    counter = mp.Value('l', 0)
    training_done = mp.Value('b', False)  # bool

    if agent is None:
        assert make_agent is not None
        agent = make_agent(0)

    shared_objects = extract_shared_objects_from_agent(agent)
    set_shared_objects(agent, shared_objects)

    print(logger)

    evaluator = AsyncEvaluator(
        n_runs=eval_n_runs,
        eval_frequency=eval_frequency, outdir=outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        explorer=eval_explorer)
#        explorer=eval_explorer,
#        logger=logger)

    def run_func(process_idx):
        random_seed.set_random_seed(process_idx)

        env = make_env(process_idx, test=False)
        eval_env = make_env(process_idx, test=True)
        if make_agent is not None:
            local_agent = make_agent(process_idx)
            set_shared_objects(local_agent, shared_objects)
        else:
            local_agent = agent
        local_agent.process_idx = process_idx

        def f():
            train_loop(
                process_idx=process_idx,
                counter=counter,
                agent=local_agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                training_done=training_done,
                eval_env=eval_env,
                logger=logger)

        if profile:
            import cProfile
            cProfile.runctx('f()', globals(), locals(),
                            'profile-{}.out'.format(os.getpid()))
        else:
            f()

    async.run_async(processes, run_func)

    return agent
