import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Concatenate, Input

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_norm_dense import SNDense
import random
import tensorflow_probability as tfp

class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, output_activation = "sigmoid", name = "Discriminator"):
        super().__init__(name = name)

        self.conv_layers  =  [Conv2D(32, 3, strides = 3, activation = 'relu'),
                              Conv2D(64, 3, strides = 2, activation = 'relu'),
                            Conv2D(128, 3, strides = 2, activation = 'relu'),
                            Conv2D(256, 3, strides = 2, activation = 'relu'),
                            GlobalAveragePooling2D()]
        self.act_layers  =  [Dense(64, activation = 'relu')]
        self.connect_layers  =  [Dense(128, activation = 'relu'), Dense(64, activation = 'relu')]
        self.out_layer  =  Dense(1, name = "reward", activation = output_activation)

        dummy_state  =  tf.constant(np.zeros(shape = (1,) + state_shape, dtype = np.float32))       # (1, 80, 80, 9), 创建tensor方法
        dummy_action  =  tf.constant(np.zeros(shape = (1, action_dim), dtype = np.float32))         # (1, 2)

        self(dummy_state, dummy_action)
        #self.summary()

    def call(self, state, action):
        features  =  state                                              # (x, 80, 80, 9)

        for conv_layer in self.conv_layers:                             # (x, 256)
            features  =  conv_layer(features)

        action  =  self.act_layers[0](action)                           # (x, 64)
        features_action  =  tf.concat([features, action], axis = 1)     # (x, 320)

        for connect_layer in self.connect_layers:
            features_action  =  connect_layer(features_action)

        values  =  self.out_layer(features_action)

        return values                                                   # (x, 1)

    def compute_reward(self, state, action):            # state = (1/32/x, 80, 80, 9)  action= (1/32/x, 2)
        return tf.math.log(self(state, action) + 1e-8)          #  shape=(1/32/?, 1)

# define imitation learning actor
def Generator(state_shape, action_dim, name = 'gen'):
    tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed

    obs  =  Input(shape = state_shape)
    conv_1  =  Conv2D(16, 3, strides = 3, activation = 'relu')(obs)
    conv_2  =  Conv2D(64, 3, strides = 2, activation = 'relu')(conv_1)
    conv_3  =  Conv2D(128, 3, strides = 2, activation = 'relu')(conv_2)
    conv_4  =  Conv2D(256, 3, strides = 2, activation = 'relu')(conv_3)
    info  =  GlobalAveragePooling2D()(conv_4)
    dense_1  =  Dense(128, activation = 'relu')(info)
    dense_2  =  Dense(32, activation = 'relu')(dense_1)
    mean  =  Dense(action_dim, activation = 'linear')(dense_2)
    std  =  Dense(action_dim, activation = 'softplus')(dense_2)

    model  =  tf.keras.Model(obs, [mean, std], name = name)         # 将定义好的网络结构封装入一个对象，用于训练、测试和预测。

    return model


class GaussianActor(tf.keras.Model):
    LOG_STD_CAP_MAX  =  2                               # np.e**2  =  7.389
    LOG_STD_CAP_MIN  =  -20                             # np.e**-10  =  4.540e-05
    EPS  =  1e-6

    def __init__(self, state_shape, action_dim, max_action, squash = False, name = 'gaussian_policy'):
        super().__init__(name = name)

        self._squash  =  squash
        self._max_action  =  max_action

        obs  =  tf.keras.layers.Input(shape = state_shape)              # (None, 80, 80, 9)

        conv_1  =  tf.keras.layers.Conv2D(32, 3, strides = 3, activation = 'relu')(obs)         # (None, 26, 26, 16)
        conv_2  =  tf.keras.layers.Conv2D(64, 3, strides = 2, activation = 'relu')(conv_1)      # (None, 12, 12, 64)
        conv_3  =  tf.keras.layers.Conv2D(128, 3, strides = 2, activation = 'relu')(conv_2)     # (None, 5, 5, 128)
        conv_4  =  tf.keras.layers.Conv2D(256, 3, strides = 2, activation = 'relu')(conv_3)     # (None, 2, 2, 256)
        info  =  tf.keras.layers.GlobalAveragePooling2D()(conv_4)                               # info = (None, 256)，全局平均池化

        dense_1  =  tf.keras.layers.Dense(128, activation = 'relu')(info)                       # (None, 128)
        dense_2  =  tf.keras.layers.Dense(64, activation = 'relu')(dense_1)                     # (None, 32)
        mean  =  tf.keras.layers.Dense(action_dim, name = "L_mean")(dense_2)                    # (None, 2)
        log_std  =  tf.keras.layers.Dense(action_dim, name = "L_logstd")(dense_2)               # (None, 2)

        self.network  =  tf.keras.Model(obs, [mean, log_std], name = 'policy_net')  # 第一个参数为输入，第二个为输出
        # self.network.summary()                                                      # 输出模型各层的参数

    def _compute_dist(self, states):
        mean, log_std  =  self.network(states)
        log_std  =  tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = tf.exp(log_std))

    def call(self, states, test = False):           # states = (32/1/?, 80, 80, 9)

        """ Compute actions and log probabilities of the selected action """
        dist  =  self._compute_dist(states)
        entropy  =  dist.entropy()

        if test:                                        # false
            raw_actions  =  dist.mean()
        else:
            raw_actions  =  dist.sample()               # (32/1/?, 2)

        log_pis  =  dist.log_prob(raw_actions)          # (32/1/?,)

        if self._squash:                                # True
            actions  =  tf.tanh(raw_actions)            # 把动作归一化到[-1,1]之间, (32/1/?, 2)
            diff  =  tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis = 1)  # (32/1/?,)
            log_pis -=  diff
            log_pis = tf.clip_by_value(log_pis, -10, 0)
        else:
            actions  =  raw_actions

        actions  =  actions * self._max_action                         # (32/1/?, 2)

        return actions, log_pis, entropy                               # (32/1/?, 2) (32/1/?, ) (32/1/?, )

class GAIL_1(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim,
            max_action = 1.0,
            lr = 2e-4,
            n_training = 1,
            horizon = 512,
            name = "GAIL",
            **kwargs):
        super().__init__(name = name, n_training = n_training, **kwargs)

        # set up discriminator
        self.disc  =  Discriminator(state_shape = state_shape, action_dim = action_dim)
        self.disc_optimizer  =  tf.keras.optimizers.Adam(learning_rate = lr, clipnorm = 5.0)
        self.horizon = horizon

        # set up generator
        # self.gen  =  Generator(state_shape = state_shape, action_dim = action_dim, name = 'gen')
        self.gen  =  Generator(state_shape, action_dim, max_action, squash = True)
        self.gen_optimizer  =  tf.keras.optimizers.Adam(learning_rate = lr, clipnorm = 5.0)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
        dis_loss, gen_loss, accuracy, js_divergence, inference_reward, real_logits, fake_logits  =  self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)

        tf.summary.scalar(name = self.policy_name + "/DiscriminatorLoss", data = dis_loss)
        tf.summary.scalar(name = self.policy_name + "/GeneratorLoss", data = gen_loss)
        tf.summary.scalar(name = self.policy_name + "/Accuracy", data = accuracy)
        tf.summary.scalar(name = self.policy_name + "/JSdivergence", data = js_divergence)
        tf.summary.scalar(name = self.policy_name + "/InferenceReward", data = inference_reward)
        tf.summary.scalar(name = self.policy_name + "/RealLogits", data = real_logits)
        tf.summary.scalar(name = self.policy_name + "/FakeLogits", data = fake_logits)


    def _compute_js_divergence(self, fake_logits, real_logits):

        p = tf.clip_by_value(fake_logits, 1e-8, 1)
        q = tf.clip_by_value(real_logits, 1e-8, 1)
        m = (p + q)

        log_p = tf.clip_by_value(p/m, 1e-8, 10)
        log_q = tf.clip_by_value(q/m, 1e-8, 10)

        return 0.5 * tf.reduce_mean(p * tf.math.log(log_p) + q * tf.math.log(log_q)) + tf.math.log(2.0)

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon  =  1e-8                            # ε

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logits  =  self.disc(expert_states, expert_acts)       # 判别器打分
                fake_logits  =  self.disc(agent_states, agent_acts)         # 判别器打分


                dis_loss  =  -(tf.reduce_mean(tf.math.log(tf.clip_by_value(real_logits, epsilon, 1))) +
                         tf.reduce_mean(tf.math.log(tf.clip_by_value(1.0 - fake_logits, epsilon, 1))))

                gen_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(fake_logits, epsilon, 1)))

            dis_grads  =  tape.gradient(dis_loss, self.disc.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(dis_grads, self.disc.trainable_variables))

            gen_grads  =  tape.gradient(gen_loss, self.disc.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.disc.trainable_variables))

        accuracy  =  (tf.reduce_mean(tf.cast(real_logits >=  0.5, tf.float32)) / 2.0 +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.0)

        js_divergence  =  self._compute_js_divergence(fake_logits, real_logits)     # js散度，计算两个概率分布之间的相似性，相同为0，相反为1
        inference_reward  =  self.disc.compute_reward(agent_states, agent_acts)

        return dis_loss, gen_loss, accuracy, js_divergence, tf.reduce_mean(inference_reward), tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

    def inference(self, states, actions, next_states):
        states  =  np.expand_dims(states, axis = 0)
        actions  =  np.expand_dims(actions, axis = 0)

        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        with tf.device(self.device):
            return self.disc.compute_reward(states, actions)

    @staticmethod
    def get_argument(parser = None):
        parser  =  IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action = 'store_true')

        return parser

    def get_action(self, state, test = False):      # state = (80, 80, 9)
        is_single_input = state.ndim == self._state_ndim

        if is_single_input:         # True
            state = np.expand_dims(state, axis = 0).astype(np.float32)      # (1, 80, 80, 9)
        print('7777')
        dist = self._compute_dist(state)
        print('888880')
        if test:                                        # false
            raw_actions  =  dist.mean()
        else:
            raw_actions  =  dist.sample()               # (32/1/?, 2)

        actions  =  tf.tanh(raw_actions)
        actions  =  actions * self._max_action
        return actions.numpy()

    @tf.function
    def _compute_dist(self, state):
       return self.gen(state)