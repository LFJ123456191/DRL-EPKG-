import os
import time

import numpy as np
import tensorflow as tf
import csv
import datetime

from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete


class IRLTrainer(Trainer):
    def __init__(
            self,
            policy,
            env,
            args,
            irl,
            expert_obs,
            expert_next_obs,
            expert_act,
            test_env = None):
        self._irl = irl
        args.dir_suffix = self._irl.policy_name + args.dir_suffix           # GAIL
        super().__init__(policy, env, args, test_env)

        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        self._random_range = range(expert_obs.shape[0])

    def __call__(self):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(self._policy, self._env)

        # Prepare local buffer
        kwargs_local_buf = get_default_rb_dict(size = self._policy.horizon, env = self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        # set up variables
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        total_steps = np.array(0, dtype = np.int32)
        n_episode = 0
        success_log = [0]
        episode_returns = []
        best_train = -np.inf
        best_success = -np.inf

        # reset env
        obs = self._env.reset()
        obs = obs[self._env.agent_id]

        # start interaction
        tf.summary.experimental.set_step(total_steps)

        while total_steps <= self._max_steps:
            ##### Collect samples #####

            for _ in range(self._policy.horizon):           # 512
                if self._normalize_obs:                     # false
                    obs = self._obs_normalizer(obs, update = False)

                # get policy actions
                act, logp, val = self._policy.get_action_and_val(obs)       # 返回动作、概率及状态价值,(2,)

                # bound actions
                if not is_discrete(self._env.action_space):             # 是连续，执行
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act

                # roll out a step
                next_obs, reward, done, info = self._env.step({self._env.agent_id: env_act})
                next_obs = next_obs[self._env.agent_id]
                reward = reward[self._env.agent_id]
                done = done[self._env.agent_id]
                info = info[self._env.agent_id]

                episode_steps += 1
                total_steps += 1
                episode_return += reward

                done_flag = done
                if (hasattr(self._env, "_max_episode_steps") and episode_steps == self._env._max_episode_steps):
                    done_flag = False

                # add a sampled step to local buffer
                reward = self._irl.inference(obs, act, next_obs)
                self.local_buffer.add(obs = obs, act = act, next_obs = next_obs, rew = reward, done = done_flag, logp = logp, val = val)
                obs = next_obs

                # add to training log
                if total_steps % 5 == 0:
                    success = np.sum(success_log[-20:]) / 20
                    with open(self._output_dir + '/training_log.csv', 'a', newline = '') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([n_episode, total_steps, episode_returns[n_episode-1] if episode_returns else -1, success, episode_steps])

                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)

                    # if the task is successful
                    success_log.append(1 if info['env_obs'].events.reached_goal else 0)
                    success = np.sum(success_log[-20:]) / 20
                    # end this eposide
                    self.finish_horizon()
                    obs = self._env.reset()
                    obs = obs[self._env.agent_id]
                    n_episode += 1
                    episode_returns.append(episode_return)
                    fps = episode_steps / (time.time() - episode_start_time)

                    # log information
                    self.logger.info("Total_Episode = {0}, Total_Steps = {1}, Episode_Steps = {2}, Return = {3: 5.4f}, FPS = {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))

                    tf.summary.scalar(name = "Common/training_return", data = episode_return)
                    tf.summary.scalar(name = "Common/training_episode_length", data = episode_steps)
                    tf.summary.scalar(name = "Common/fps", data = fps)
                    tf.summary.scalar(name = 'Common/training_success', data = success)

                    # reset variables
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                    time_format='%m%dT%H%M'
                    data_time =  datetime.datetime.now().strftime(time_format)
                    # print('模型参数 =')
                    # self._policy.actor.network.summary()
                    # save policy model
                    # if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
                    #     train = np.mean(episode_returns[-20:])
                    #     self._policy.actor.network.save('{}/Model/Model_{}_{:.4f}.h5'.format(self._logdir, n_episode, train))

                    if n_episode > 20:
                        if np.mean(episode_returns[-20:]) >= best_train:
                            best_train  =  np.mean(episode_returns[-20:])
                            self._policy.actor.network.save('{}/Model/{}_{}_{:.3f}_{}.h5'.format(self._logdir, data_time, n_episode, best_train, success))
                            if success >= best_success:
                                best_success  =  success
                        elif success > best_success or success >= 0.7:
                            best_success  =  success
                            self._policy.actor.network.save('{}/Model/{}_{}_{:.3f}_{}.h5'.format(self._logdir, data_time, n_episode, best_train, success))
                # test the policy
                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes))
                    tf.summary.scalar(name = "Common/average_test_return", data = avg_test_return)
                    tf.summary.scalar(name = "Common/average_test_episode_length", data = avg_test_steps)
                    self.writer.flush()

                    obs = self._env.reset()
                    obs = obs[self._env.agent_id]
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.perf_counter()

                # save model
                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            tf.summary.experimental.set_step(total_steps)

            ##### Train generator and discriminator ####
            if self._policy.normalize_adv:                              # True
                samples = self.replay_buffer.get_all_transitions()      # dict_keys(['obs', 'act', 'done', 'logp', 'ret', 'adv'])
                mean_adv = np.mean(samples["adv"])                  # samples["adv"] = (512, 1)
                std_adv = np.std(samples["adv"])
                # Update normalizer
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])

            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):          # 128
                 # train generator

                for _ in range(self._policy.n_epoch):                   # 10
                    samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))  # 随机排序

                    # normalize observation
                    if self._normalize_obs:
                        samples["obs"] = self._obs_normalizer(samples["obs"], update = False)

                    # normalize advantage
                    if self._policy.normalize_adv:      # True
                        adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)            # 归一化
                    else:
                        adv = samples["adv"]

                    # policy improvement
                    for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                        target = slice(idx * self._policy.batch_size, (idx + 1) * self._policy.batch_size)

                        self._policy.train(
                            states = samples["obs"][target],
                            actions = samples["act"][target],
                            advantages = adv[target],
                            logp_olds = samples["logp"][target],
                            returns = samples["ret"][target])

                # train discriminator
                for _ in range(self._irl.n_training):               # 1
                    for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                        target = slice(idx * self._policy.batch_size, (idx + 1) * self._policy.batch_size)

                        # Do not allow duplication
                        indices = np.random.choice(self._random_range, self._irl.batch_size, replace = False)       # _random_range = range(0, 8898)
                        self._irl.train(
                            agent_states = samples["obs"][target],
                            agent_acts = samples["act"][target],
                            agent_next_states = None,
                            expert_states = self._expert_obs[indices],
                            expert_acts = self._expert_act[indices],
                            expert_next_states = self._expert_next_obs[indices])

        tf.summary.flush()

    def finish_horizon(self, last_val = 0):
        self.local_buffer.on_episode_end()
        samples = self.local_buffer._encode_sample(np.arange(self.local_buffer.get_stored_size()))

        # add the value of the last step if any
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]

        if self._policy.enable_gae:                                                     # True
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)    # (Episode_Steps,)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]           # (Episode_Steps,)

        self.replay_buffer.add(
            obs = samples["obs"], act = samples["act"], done = samples["done"],
            ret = rets, adv = advs, logp = np.squeeze(samples["logp"]))

        # clear local buffer
        self.local_buffer.clear()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        avg_test_steps = 0

        if self._save_test_path:                        #
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size = self._episode_max_steps)

        for i in range(self._test_episodes):            # 20
            episode_return = 0.
            obs = self._test_env.reset()
            obs = obs[self._test_env.agent_id]
            avg_test_steps += 1

            for _ in range(self._episode_max_steps):             # 1000
                if self._normalize_obs:                             # false
                    obs = self._obs_normalizer(obs, update = False)

                act, _ = self._policy.get_action(obs, test = True)      # 返回高斯的动作、概率
                act = (act if is_discrete(self._env.action_space) else
                       np.clip(act, self._env.action_space.low, self._env.action_space.high))

                next_obs, reward, done, _ = self._test_env.step({self._test_env.agent_id: act})
                next_obs = next_obs[self._test_env.agent_id]
                reward = reward[self._test_env.agent_id]
                done = done[self._test_env.agent_id]
                avg_test_steps += 1
                print('self._save_test_path =', self._save_test_path)
                if self._save_test_path:                    #
                    replay_buffer.add(obs = obs, act = act, next_obs = next_obs, rew = reward, done = done)

                episode_return += reward
                obs = next_obs

                if done:
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes
