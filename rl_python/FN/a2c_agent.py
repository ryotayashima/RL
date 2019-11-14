import argparse
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras as K
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer
tf.compat.v1.disable_eager_execution()


class ActorCriticAgent(FNAgent):

  def __init__(self, actions):
    # ActorCriticAgent uses self policy (doesn't use epsilon).
    super().__init__(epsilon=0.0, actions=actions)
    self._updater = None

  @classmethod
  def load(cls, env, model_path):
    actions = list(range(env.action_space.n))
    agent = cls(actions)
    agent.model = K.models.load_model(model_path, custom_objects={     
                                      "SampleLayer": SampleLayer})
    agent.initialized = True
    return agent

  def initialize(self, experiences, optimizer):
    feature_shape = experiences[0].s.shape
    self.make_model(feature_shape)
    self.set_updater(optimizer)
    self.initialized = True
    print("Done initialization. From now, begin training!")

  def make_model(self, feature_shape):
    normal = K.initializers.glorot_normal()
    model = K.Sequential()
    model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape,
            kernel_initializer=normal, activation="relu"))
    model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal, activation="relu"))
    model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal, activation="relu"))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))

    actor_layer = K.layers.Dense(len(self.actions),
                                  kernel_initializer=normal)
    action_evals = actor_layer(model.output)
    actions = SampleLayer()(action_evals)

    critic_layer = K.layers.Dense(1, kernel_initializer=normal)
    values = critic_layer(model.output)

    self.model = K.Model(inputs=model.input,
                          outputs=[actions, action_evals, values])

  def set_updater(self, optimizer,
                  value_loss_weight=1.0, entropy_weight=0.1):
    actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
    values = tf.compat.v1.placeholder(shape=(None), dtype="float32")

    _, action_evals, estimates = self.model.output

    neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=action_evals, labels=actions)
    # tf.stop_gradient: Prevent policy_loss influences critic_layer.
    advantages = values - tf.stop_gradient(estimates)

    policy_loss = tf.reduce_mean(neg_logs * advantages)
    value_loss = K.losses.MeanSquaredError()(values, estimates)
    action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))

    loss = policy_loss + value_loss_weight * value_loss
    loss -= entropy_weight * action_entropy

    updates = optimizer.get_updates(loss=loss, 
                                    parms=self.model.trainable_weights)
    
    self._updater = K.backend.function(
      inputs=[self.model.input,
              actions, values],
      outputs=[loss,
               policy_loss,
               value_loss,
               tf.reduce_mean(neg_logs),
               tf.reduce_mean(advantages),
               action_entropy],
      updates=updates)

  def categorical_entropy(self, logits):
    """
    From OpenAI baseline implementation.
    https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L192
    """
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

  def policy(self, s):
    if not self.initialized:
      return np.random.randint(len(self.actions))
    else:
      action, action_evals, values = self.model.predict(np.array([s]))
      return action[0]

  def estimate(self, s):
    action, action_evals, values = self.model.predict(np.array([s]))
    return values[0][0]

  def update(self, states, actions, rewards):
    return self._updater([states, actions, rewards])


class SampleLayer(K.layers.Layer):

  def __init__(self, **kwards):
    self.output_dim = 1   # sample one action from evaluations
    super(SampleLayer, self).__init__(**kwards)

  def build(self, input_shape):
    super(SampleLayer, self).build(input_shape)

  def call(self, x):
    noise = tf.random.uniform(tf.shape(x))
    return tf.argmax(x - tf.math.log(-tf.math.log(noise)), axis=1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)


class ActorCriticAgentTest(ActorCriticAgent):

  def make_model(self, feature_shape):
    normal = K.initializers.glorot_normal()
    model = K.Sequential()
    model.add(K.layers.Dense(10, input_shape=feature_shape,
                             kernel_initializer=normal, activation="relu"))
    model.add(K.layers.Dense(10, kernel_initializer=normal,
                              activation="relu"))

    actor_layer = K.layers.Dense(len(self.actions),
                                  kernel_initializer=normal)

    action_evals = actor_layer(model.output)
    actions = SampleLayer()(action_evals)

    critic_layer = K.layers.Dense(1, kernel_initializer=normal)
    values = critic_layer(model.output)

    self.model = K.Model(inputs=model.input,
                         outputs=[actions, action_evals, values])
    
   
class CatcherObserver(Observer):

  def __init__(self, env, width, height, frame_count):
    super().__init__(env)
    self.width = width
    self.height = height
    self.frame_count = frame_count
    self._frames = deque(maxlen=frame_count)

  def transform(self, state):
    grayed = Image.fromarray(state).convert("L")
    resized = grayed.resize((self.width, self.height))
    resized = np.array(resized).astype("float")
    normalized = resized / 255.0
    if len(self._frames) == 0:
      for i in range(self.frame_count):
        self._frames.append(normalized)
      else:
        self._frames.append(normalized)
    feature = np.array(self._frames)
    # Convert the feature shape (f, w, h) => (h, w, f).
    feature = np.transpose(feature, (1, 2, 0))
    return feature


class ActorCriticTrainer(Trainer):

  def __init__(self, buffer_size=256, batch_size=32,
               gamma=0.99, learning_rate=1e-3,
               report_interval=10, log_dir='', file_name=''):
    super().__init__(buffer_size, batch_size, gamma,
                     report_interval, log_dir)
    self.file_name = file_name if file_name else "a2c_agent.h5"
    self.learning_rate = learning_rate
    self.losses = {}
    self.rewards = []
    self._max_reward = -10

  