import os
import time

import numpy as np
import tensorflow as tf
import csv

from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete

class GAIL_Trainer(Trainer):
    def __init__(
            self,
            idx,
            gail,
            env,
            args,
            expert_obs,
            expert_next_obs,
            expert_act,
            test_env = None):

        self.idx = idx
        self.gail = gail
        self.file_path = args.file_path
        self.samples = args.samples
        self.horizon = args.horizon

        super().__init__(gail, env, args, test_env)

        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        self._random_range = range(expert_obs.shape[0])


    def __call__(self):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(self.gail, self._env)

        # Prepare local buffer
        kwargs_local_buf = get_default_rb_dict(size = self.horizon, env = self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        if not os.path.exists('./gail_model_2/{}_{}'.format(self.file_path.split('/')[-1], self.samples)):
            os.makedirs('./gail_model_2/{}_{}'.format(self.file_path.split('/')[-1], self.samples))

        # set up variables
        episode_steps = 0
        total_steps = np.array(0, dtype = np.int32)
        n_episode = 0
        success_log = [0]
        best_success = -np.inf


        # start interaction
        tf.summary.experimental.set_step(total_steps)
        n = 0
        # sample actions
        while total_steps <= len(self._expert_obs)-1:
            obs = self._expert_obs[total_steps]

            for _ in range(self.horizon):           # 512

                if(n % 10 == 0):
                    print('total_steps =', total_steps)
                n = n + 1

                # get policy actions
                act = self.gail.get_action(obs)       #

                # bound actions
                if not is_discrete(self._env.action_space):             #
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act
                print('0000')
                # roll out a step
                next_obs, reward, done, info = self._env.step({self._env.agent_id: env_act})
                next_obs = next_obs[self._env.agent_id]
                reward = reward[self._env.agent_id]
                done = done[self._env.agent_id]
                info = info[self._env.agent_id]

                episode_steps += 1
                total_steps += 1
                print('1111')
                done_flag = done
                if (hasattr(self._env, "_max_episode_steps") and episode_steps == self._env._max_episode_steps):
                    done_flag = False

                # add a sampled step to local buffer

                self.local_buffer.add(obs = obs, act = env_act, done = done_flag)
                obs = self._expert_obs[total_steps]
                print('2222')
                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)

                    # if the task is successful
                    success_log.append(1 if info['env_obs'].events.reached_goal else 0)
                    success = np.sum(success_log[-20:]) / 20

                    # end this eposide
                    self.finish_horizon()
                    print('3333')
                    n_episode += 1

                    if n_episode > 20 and success > best_success:
                        best_success = success
                        print('best_success =', best_success)
                        self._policy.actor.network.save('./gail_model_2/{}_{}/ensemble_{}.h5'.format(self.file_path.split('/')[-1], self.samples, self.idx))
                    # reset variables
                    episode_steps = 0

            tf.summary.experimental.set_step(total_steps)

            ##### Train generator and discriminator ####
            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):          # 128
                 # train generator
                print('5555')
                for _ in range(self._policy.n_epoch):                   # 10
                    samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))  #

                    for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                        target = slice(idx * self._policy.batch_size, (idx + 1) * self._policy.batch_size)
                        indices = np.random.choice(self._random_range, self.gail.batch_size, replace = False)
                        self.gail.train(
                            agent_states = samples["obs"][target],
                            agent_acts = samples["act"][target],
                            expert_states = self._expert_obs[indices],
                            expert_acts = self._expert_act[indices])
        tf.summary.flush()

    def finish_horizon(self, last_val = 0):
        self.local_buffer.on_episode_end()
        samples = self.local_buffer._encode_sample(np.arange(self.local_buffer.get_stored_size()))

        # add the value of the last step if any
        rews = np.append(samples["rew"], last_val)

        self.replay_buffer.add(
            obs = samples["obs"], act = samples["act"], done = samples["done"])

        # clear local buffer
        self.local_buffer.clear()
