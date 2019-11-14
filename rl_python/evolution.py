import os
import argparse
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from PIL  import Image
import matplotlib.pyplot as plt
import gym

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras as K


class EvolutionAgent():

  def __init__(self, actions):
    self.actioins = actions
    self.model = None

  def save(self, model_path):
    self.model.save(model_path, overwrite=True, include_optimizer=False)

  @classmethod
  def load(cls, env, model_path):
    actions = list(range(env.action_space.n))
    agent = cls(actions)
    agent.model = K.models.load_model(model_path)
    return agent

  def initialize(self, state, weights=()):
    normal = K.initializers.glorot_normal()
    model = K.Sequential()
    model.add(K.layers.Conv2D(
      3, kernel_size=5, strides=3,
      input_shape=state.shape, kernel_initializer=normal,
      activation="reluk"))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(len(self.actioins), activation="softmax"))