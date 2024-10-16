import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Input
from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.tfp_gaussian_actor import GaussianActor
import random
import tensorflow_probability as tfp
# from scipy.stats import wasserstein_distance

class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.conv_layers = [Conv2D(32, 3, strides = 3, activation = 'relu'),
                            Conv2D(64, 3, strides = 2, activation='relu'),
                            Conv2D(128, 3, strides = 2, activation = 'relu'),
                            Conv2D(256, 3, strides = 2, activation = 'relu'),
                            GlobalAveragePooling2D()]
        self.connect_layers = [Dense(128, activation = 'relu'), Dense(64, activation = 'relu')]
        self.out_layer = Dense(1, name = "V", activation = 'linear')

        dummy_state = tf.constant(np.zeros(shape = (1,) + state_shape, dtype = np.float32))
        self(dummy_state)
        #self.summary()

    def call(self, states):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        for connect_layer in self.connect_layers:
            features = connect_layer(features)

        values = self.out_layer(features)


        return tf.squeeze(values, axis = 1)


# define imitation learning actor
def Actor(state_shape, action_dim = 2, name = 'actor'):
    tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed

    obs  =  Input(shape = state_shape)
    conv_1  =  Conv2D(32, 3, strides = 3, activation = 'relu')(obs)
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


def wasserstein_distance(logp_news, logp_olds):
    loss = tf.reduce_mean(tf.abs(logp_news - logp_olds))
    return loss

class PPO(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            max_action = 1.0,
            lr_actor = 2e-4,
            lr_critic = 2e-4,
            clip = True,
            clip_ratio = 0.2,
            name = "PPO",
            **kwargs):
        super().__init__(name = name, **kwargs)

        self.clip  =  clip
        self.clip_ratio  =  clip_ratio
        self._max_action  =  max_action

        # set up actor and critic
        self.actor = GaussianActor(state_shape, action_dim, max_action, squash = True)
        # self.actor = Actor(state_shape)
        self.critic = CriticV(state_shape)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_actor, clipnorm = 3)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_critic, clipnorm = 3)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]


    def get_action(self, state, test = False):
        msg = "Input instance should be np.ndarray, not {}".format(type(state))
        assert isinstance(state, np.ndarray), msg

        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis = 0).astype(np.float32)

        action, logp, entropy = self._get_action_body(state, test)

        if is_single_input:     # True
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_1(self, state, test = False):      # state = (80, 80, 9)
        is_single_input = state.ndim == self._state_ndim

        if is_single_input:         # True
            state = np.expand_dims(state, axis = 0).astype(np.float32)      # (1, 80, 80, 9)

        dist = self._get_action_body_1(state, test)

        if test:                                        # false
            raw_actions  =  dist.mean()
        else:
            raw_actions  =  dist.sample()               # (32/1/?, 2)

        actions  =  tf.tanh(raw_actions)
        actions  =  actions * self._max_action
        return actions.numpy()


    def get_action_and_val(self, state, test = False):      # state = (80, 80, 9)
        is_single_input = state.ndim == self._state_ndim

        if is_single_input:         # True
            state = np.expand_dims(state, axis = 0).astype(np.float32)      # (1, 80, 80, 9)

        action, logp, v = self._get_action_logp_v_body(state, test)

        if is_single_input:     # True
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    @tf.function
    def _get_action_logp_v_body(self, state, test):
        action, logp, entropy = self.actor(state, test)
        v = self.critic(state)

        return action, logp, v


    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)

    @tf.function
    def _get_action_body_1(self, state, test):
        mean, std  =  self.actor(state, test)
        return tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = std)


    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        actor_loss, logp_news, ratio, ent  =  self._train_actor_body(states, actions, advantages, logp_olds)
        critic_loss  =  self._train_critic_body(states, returns)

        # Visualize results in TensorBoard
        tf.summary.scalar(name = self.policy_name + "/actor_loss", data = actor_loss)
        tf.summary.scalar(name = self.policy_name + "/logp_max", data = np.max(logp_news))
        tf.summary.scalar(name = self.policy_name + "/logp_min", data = np.min(logp_news))
        tf.summary.scalar(name = self.policy_name + "/logp_mean", data = np.mean(logp_news))
        tf.summary.scalar(name = self.policy_name + "/adv_max", data = np.max(advantages))
        tf.summary.scalar(name = self.policy_name + "/adv_min", data = np.min(advantages))
        tf.summary.scalar(name = self.policy_name + "/kl", data = tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name = self.policy_name + "/entropy", data = ent)
        tf.summary.scalar(name = self.policy_name + "/ratio", data = tf.reduce_mean(ratio))
        tf.summary.scalar(name = self.policy_name + "/critic_loss", data = critic_loss)

        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):            # advantages = (32/？, 1)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                ent  =  tf.reduce_mean(self.actor.compute_entropy(states))          # 计算熵，熵的均值

                if self.clip:                                                       # True
                    logp_news  =  self.actor.compute_log_probs(states, actions)     # 新策略的概率对数
                    logp_news  =  tf.clip_by_value(logp_news, -10, 10)              # (32,)

                    ratio  =  tf.math.exp(logp_news - tf.squeeze(logp_olds))        # 重要性权重
                    min_adv  =  tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * tf.squeeze(advantages)      # 根据经验clip_ratio取0.1和0.2时实际效果较好
                    actor_loss  =  -tf.reduce_mean(tf.minimum(ratio * tf.squeeze(advantages), min_adv))

                    actor_loss -=  self.entropy_coef * ent          # self.entropy_coef = 0.01

                    # logp_olds = tf.squeeze(logp_olds)
                    # wasserstein_loss = wasserstein_distance(logp_news, logp_olds)

                else:
                    raise NotImplementedError

            # Update actor
            actor_grad  =  tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            # actor_grad  =  tape.gradient(wasserstein_loss, self.actor.trainable_variables)
            # self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return actor_loss, logp_news, ratio, ent
        # return wasserstein_loss, logp_news, ratio, ent

    @tf.function
    def _train_critic_body(self, states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                current_V = self.critic(states)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(tf.square(td_errors))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
