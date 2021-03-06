from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import logging
import os
import numpy as np

from chainerrl.experiments.evaluator import Evaluator
from chainerrl.experiments.evaluator import save_agent
from chainerrl.misc.ask_yes_no import ask_yes_no
from chainerrl.misc.makedirs import makedirs

from chainer import variable
import chainer.cuda as xp


def save_agent_replay_buffer(agent, t, outdir, suffix='', logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, '{}{}.replay.pkl'.format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info('Saved the current replay buffer to %s', filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=''):
    if hasattr(agent, 'replay_buffer') and \
            ask_yes_no('Replay buffer has {} transitions. Do you save them to a file?'.format(len(agent.replay_buffer))):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def train_agent(agent, env, steps, outdir, max_episode_len=None,
                step_offset=0, evaluator=None, successful_score=None,
                logger=None):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()
    r = 0
    done = False

    t = step_offset
    agent.t = step_offset

    episode_len = 0

    action = 0
    action_array = variable.Variable(xp.zeros((1, env.number_of_actions), dtype=np.float32))


    while t < steps:
        # added by SM
        action_array.data.fill(0)
        action_array.data[0][action]=1
        # a_t
        #action = agent.act_and_train(obs, r)
        action = agent.act_and_train(obs, r, action_array)
        # o_{t+1}, r_{t+1}
        obs, r, done, info = env.step(action)
        t += 1
        episode_r += r
        episode_len += 1

        if done or episode_len == max_episode_len or t == steps:
            agent.stop_episode_and_train(obs, r, done=done)
            logger.info('outdir:%s step:%s episode:%s R:%s',
                        outdir, t, episode_idx, episode_r)
            logger.info('statistics:%s', agent.get_statistics())
            if evaluator is not None:
                evaluator.evaluate_if_necessary(t)
                if (successful_score is not None and
                        evaluator.max_score >= successful_score):
                    break
            if t == steps:
                break
            # Start a new episode
            episode_r = 0
            episode_idx += 1
            episode_len = 0
            obs = env.reset()
            r = 0
            done = False

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')


def train_agent_with_evaluation(
        agent, env, steps, eval_n_runs, eval_frequency,
        outdir, max_episode_len=None, step_offset=0, eval_explorer=None,
        eval_max_episode_len=None, eval_env=None, successful_score=None,
        render=False, logger=None):
    """Run a DQN-like agent.

    Args:
      agent: Agent.
      env: Environment.
      steps (int): Number of total time steps for training.
      eval_n_runs (int): Number of runs for each time of evaluation.
      eval_frequency (int): Interval of evaluation.
      outdir (str): Path to the directory to output things.
      max_episode_len (int): Maximum episode length.
      step_offset (int): Time step from which training starts.
      eval_explorer: Explorer used for evaluation.
      eval_env: Environment used for evaluation.
      successful_score (float): Finish training if the mean score is greater
          or equal to this value if not None
    """

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
        n_runs=eval_n_runs, eval_frequency=eval_frequency,
        outdir=outdir, max_episode_len=eval_max_episode_len,
        explorer=eval_explorer, env=eval_env, step_offset=step_offset)
#                          step_offset=step_offset,
#                          logger=logger)

    train_agent(
        agent, env, steps, outdir, max_episode_len=max_episode_len,
        step_offset=step_offset, evaluator=evaluator,
        successful_score=successful_score, logger=logger)
