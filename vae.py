import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import sys
import argparse
import os
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Input, Conv2DTranspose

# read args
parser  =  argparse.ArgumentParser()
parser.add_argument('file_path')                                    # expert_data/left_turn
parser.add_argument('--samples', type = int, default = 40)
args  =  parser.parse_args()

# GPU setting
gpus  =  tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# create directory
if not os.path.exists('./expert_model/{}_{}'.format(args.file_path.split('/')[-1], args.samples)):
    os.makedirs('./expert_model/{}_{}'.format(args.file_path.split('/')[-1], args.samples))


def noise_reparameterize(mean, std):
    eps = tf.random.normal(tf.shape(mean))
    z = mean + eps * tf.math.exp(std)
    return z

def VAE(state_shape, action_dim, name = 'actor'):
    tf.random.set_seed(random.randint(1, 1000))                             # reset tensorflow random seed
    obs  =  Input(shape = state_shape)                                      # (None, 80, 80, 9)
    conv_1  =  Conv2D(16, 3, strides = 3, activation = 'relu')(obs)         # (None, 26, 26, 16)
    conv_2  =  Conv2D(64, 3, strides = 2, activation = 'relu')(conv_1)      # (None, 12, 12, 64)
    conv_3  =  Conv2D(256, 3, strides = 2, activation = 'relu')(conv_2)     # (None, 5, 5, 128)
    info  =  GlobalAveragePooling2D()(conv_3)                               # info = (None, 256)

    dense_1  =  Dense(128, activation = 'relu')(info)                       # (None, 128)
    dense_2  =  Dense(64, activation = 'relu')(dense_1)                     # (None, 32)
    mean  =  Dense(action_dim, name = "L_mean")(dense_2)                    # (None, 2)
    std  =  Dense(action_dim, activation = 'softplus')(dense_2)

    z = noise_reparameterize(mean, std)
    out_1 = Dense(64, activation = 'relu')(z)
    out_2 = Dense(128, activation = 'relu')(out_1)
    out = Dense(2, activation = 'linear')(out_2)
    model =  tf.keras.Model(obs, [out, mean, std], name = name)

    return model


EPS  =  1e-6
def loss_function(pre_y, y, mean, std):
    mse_loss =  tf.reduce_mean(tf.square(pre_y -  y))
    var = tf.square(std)
    kld_loss = 0.5 * tf.reduce_mean(tf.math.log(var + EPS) + tf.math.square(y - mean)/(var + EPS))

    return mse_loss + kld_loss

# set up ensemble
state_shape  =  (80, 80, 9)
action_dim  =  2
ensemble  =  [VAE(state_shape, action_dim, name = 'prior_{}'.format(i)) for i in range(1, 4)]


# load and process data
OBS  =  []
ACT  =  []

files  =  glob.glob(args.file_path + '/*.npz')
if args.samples < len(files):
    files  =  random.sample(files, args.samples)

for file in files:
    obs  =  np.load(file)['obs']
    act  =  np.load(file)['act']

    for i in range(obs.shape[0]):
        OBS.append(obs[i])
        act[i, 0] +=  random.normalvariate(0, 0.1)
        act[i, 0]  =  np.clip(act[i, 0], 0, 10)
        act[i, 0]  =  2.0 * ((act[i, 0] - 0) / (10 - 0)) - 1.0 # normalize speed
        act[i, 1] +=  random.normalvariate(0, 0.1)          # add a small noise to lane change, which does not affect the decision
        ACT.append(act[i])

OBS  =  np.array(OBS, dtype = np.float32)           # (x，80, 80,9)
ACT  =  np.array(ACT, dtype = np.float32)           # (x, 2)

# model training
epochs  =  100
EPS  =  1e-6

for idx, model in enumerate(ensemble):
    print(' =====  Training Ensemble Model {}  ===== '.format(idx + 1))
    tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed
    np.random.seed(random.randint(1, 1000)) # reset numpy random seed
    optimizer  =  tf.keras.optimizers.Adam(learning_rate = 3e-4) # set up optimizer

    # create dataset
    train_dataset  =  tf.data.Dataset.from_tensor_slices((OBS, ACT))    #
    train_dataset  =  train_dataset.shuffle(OBS.shape[0]).batch(32)             #
    train_loss_results  =  []

    # start training
    for epoch in range(epochs):
        epoch_loss  =  []
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                pre_y, mean, std  =  model(x)
                var  =  tf.square(std)
                loss_value = loss_function(pre_y, y, mean, std)

            epoch_loss.append(loss_value.numpy())
            gradients  =  tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss_results.append(np.mean(epoch_loss))
        sys.stdout.write("Progress: {}/{}, Loss: {:.6f}\n".format(epoch + 1, epochs, np.mean(epoch_loss)))      # 输出
        sys.stdout.flush()          #

    # save trained model
    model.save('./expert_model/{}_{}/ensemble_{}.h5'.format(args.file_path.split('/')[-1], args.samples, idx + 1), save_format = "tf")
