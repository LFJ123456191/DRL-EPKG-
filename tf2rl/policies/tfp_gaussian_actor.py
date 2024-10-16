import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Input

class GaussianActor(tf.keras.Model):
    LOG_STD_CAP_MAX  =  2
    LOG_STD_CAP_MIN  =  -20
    EPS  =  1e-6

    def __init__(self, state_shape, action_dim, max_action, squash = False, name = 'gaussian_policy'):
        super().__init__(name = name)

        self._squash  =  squash
        self._max_action  =  max_action

        obs  =  Input(shape = state_shape)

        conv_1  =  Conv2D(16, 3, strides = 3, activation = 'relu')(obs)
        conv_2  =  Conv2D(64, 3, strides = 2, activation = 'relu')(conv_1)
        conv_3  =  Conv2D(128, 3, strides = 2, activation = 'relu')(conv_2)
        conv_4  =  Conv2D(256, 3, strides = 2, activation = 'relu')(conv_3)
        info  =  GlobalAveragePooling2D()(conv_4)

        dense_1  =  Dense(128, activation = 'relu')(info)
        dense_2  =  Dense(32, activation = 'relu')(dense_1)
        mean  =  Dense(action_dim, activation = 'linear', name = "L_mean")(dense_2)
        log_std  =  Dense(action_dim,activation = 'softplus', name = "L_logstd")(dense_2)

        self.network  =  tf.keras.Model(obs, [mean, log_std], name = 'policy_net')  #
        # self.network.summary()                                                      #

    def _compute_dist(self, states):
        mean, log_std  =  self.network(states)
        log_std  =  tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = tf.exp(log_std))

    def call(self, states, test = False):           #

        """ Compute actions and log probabilities of the selected action """
        dist  =  self._compute_dist(states)
        entropy  =  dist.entropy()

        if test:                                        #
            raw_actions  =  dist.mean()
        else:
            raw_actions  =  dist.sample()               #

        log_pis  =  dist.log_prob(raw_actions)          #

        if self._squash:                                #
            actions  =  tf.tanh(raw_actions)            #
            diff  =  tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis = 1)
            log_pis -=  diff
            log_pis = tf.clip_by_value(log_pis, -10, 0)
        else:
            actions  =  raw_actions

        actions  =  actions * self._max_action                         #

        return actions, log_pis, entropy                               #

    def compute_log_probs(self, states, actions):
        raw_actions  =  actions / self._max_action
        dist  =  self._compute_dist(states)
        logp_pis  =  dist.log_prob(raw_actions)

        return logp_pis

    def compute_entropy(self, states):
        dist  =  self._compute_dist(states)
        return dist.entropy()

def wasserstein_distance(log_pis, exp_log_pis):
    wasserstein_d = tf.reduce_mean(log_pis - exp_log_pis)
    return wasserstein_d


class ExpertGuidedGaussianActor(tf.keras.Model):

    def __init__(self, state_shape, action_dim, max_action, prior, uncertainty = 'ensemble', name = 'gaussian_policy'):
        super().__init__(name = name)
        self._uncertainty  =  uncertainty
        self._max_action  =  max_action
        print('prior =', prior)
        self._expert_ensemble  =  [load_model(model) for model in glob.glob('./' + prior  + '/ensemble*.h5')]
        print('self._expert_ensemble =', self._expert_ensemble)

        # actor network
        obs  =  Input(shape = state_shape)
        conv_1  =  Conv2D(16, 3, strides = 3, activation = 'relu')(obs)
        conv_2  =  Conv2D(64, 3, strides = 2, activation = 'relu')(conv_1)
        conv_3  =  Conv2D(128, 3, strides = 2, activation = 'relu')(conv_2)
        conv_4  = Conv2D(256, 3, strides = 2, activation = 'relu')(conv_3)
        info  =  GlobalAveragePooling2D()(conv_4)

        dense_1  =  Dense(128, activation = 'relu')(info)
        dense_2  =  Dense(32, activation = 'relu')(dense_1)
        mean  = Dense(action_dim, activation = 'linear')(dense_2)
        std  =  Dense(action_dim, activation = 'softplus')(dense_2)

        self.network  =  tf.keras.Model(obs, [mean, std], name = 'RL_agent')

        dummy_state  =  tf.constant(np.zeros(shape = (1,) + state_shape, dtype = np.float32))       # ?

    def _compute_dist(self, states):
        mean, std  =  self.network(states)

        return tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = std)
    def call(self, states, test = False):
        """
        Compute actions and log probabilities of the selected action

        """
        dist  =  self._compute_dist(states)
        entropy  =  dist.entropy()

        if test:
            actions  =  dist.mean() * self._max_action
            log_pis  =  dist.log_prob(actions)

            return actions, log_pis, entropy
        else:

            actions  =  dist.sample() * self._max_action

            expert_dist, mean, std  =  self._expert_policy(states)
            log_pis  =  dist.log_prob(actions)
            exp_log_pis  =  expert_dist.log_prob(actions)

            kl = tfp.distributions.kl_divergence(dist, expert_dist)

            return actions, log_pis, entropy, std, exp_log_pis,  kl

    def _expert_policy(self, states):
        means  =  []
        variances  =  []

        for model in self._expert_ensemble:
            model.trainable  =  False
            mean, std  =  model(states)

            std +=  1e-1
            means.append(mean)
            variances.append(tf.square(std))

        if self._uncertainty  ==  'ensemble':
            mixture_mean  =  tf.reduce_mean(means, axis = 0)
            mixture_var   =  tf.reduce_mean(variances + tf.square(means), axis = 0) - tf.square(mixture_mean)
        elif self._uncertainty  ==  'policy':
            mixture_mean  =  means[0]
            mixture_var   =  variances[0]
        elif self._uncertainty  ==  'fixed':
            mixture_mean  =  means[0]
            mixture_var   =  tf.constant(0.2**2, shape = mixture_mean.shape, dtype = tf.float32)
        else:
            raise TypeError

        dist  =  tfp.distributions.MultivariateNormalDiag(loc = mixture_mean, scale_diag = tf.sqrt(mixture_var))

        return dist, mixture_mean, tf.sqrt(mixture_var)

    def compute_entropy(self, states):
        dist  =  self._compute_dist(states)

        return dist.entropy()

    def compute_log_probs(self, states, actions):
        dist  =  self._compute_dist(states)

        return dist.log_prob(actions)


    def _compute_js_divergence(self, fake_logits, real_logits):


        fake_logits = tf.clip_by_value(fake_logits, -10, 0)
        real_logits = tf.clip_by_value(real_logits, -10, 0)

        p = tf.exp(fake_logits)
        q = tf.exp(real_logits)
        m = (p + q)


        log_p = tf.clip_by_value(p/m, 1e-8, 1)
        log_q = tf.clip_by_value(q/m, 1e-8, 1)

        return 0.5 * tf.reduce_mean(p * tf.math.log(log_p) + q * tf.math.log(log_q)) + tf.math.log(2.0)