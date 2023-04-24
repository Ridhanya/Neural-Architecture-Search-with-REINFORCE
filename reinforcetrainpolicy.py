from reinforceNAS import *
import tensorflow_probability as tfp
import numpy as np
from reinforcevariables import *
import tensorflow as tf

#define optimizer
opt = tf.keras.optimizers.Adam(lr=controller_lr)

def a_loss(prob, action, reward): 
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    log_prob = dist.log_prob(action)
    loss = -log_prob*reward
    return loss

def trainpolicy(states, rewards, actions,  model, alpha=0.8):
    discounted_rewards = []

    for i in range(len(rewards)):
        cumulative_reward = 0
        for t in rewards[i:]:
            cumulative_reward += (alpha**i)*t
        discounted_rewards.append(cumulative_reward)
    

    for state, reward, action in zip(states, discounted_rewards, actions):
        with tf.GradientTape() as tape:
            #print("state ", state.shape)
            p = model(np.array(state), training=True)
            loss = a_loss(p, action, reward)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    

