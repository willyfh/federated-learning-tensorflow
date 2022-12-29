import argparse
import logging
import math
import os
import pickle
import random
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import timedelta
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
tf.get_logger().setLevel(logging.ERROR)


MODEL = None # mlp or cnn
DATA = None # iid or noniid
N_CLIENTS = None
C = None
E = None
B = None # -1 for a single batch
C_ROUNDS = None
LRS = None # list of learning rate


class MLP:
  @staticmethod
  def build(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200,  activation="relu"))
    model.add(Dense(10,  activation="softmax"))
    return model

class CNN:
  @staticmethod
  def build(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

class Range(object):
  def __init__(self, start, end):
      self.start = start
      self.end = end
  def __eq__(self, other):
      return self.start <= other <= self.end


def random_seed(seed_value=0):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
  tf.compat.v1.keras.backend.set_session(sess)



def iid_partition(y_train):
  """
    the data is shuffled, and then partitioned into 100 clients each receiving 600 examples
  """
  n_per_client = int(len(y_train)/N_CLIENTS)
  indexes_per_client = {}
  indexes = np.arange(0, len(y_train))
  random.shuffle(indexes)
  for i in range(N_CLIENTS):
    start_idx = i*n_per_client
    indexes_per_client[i] = indexes[start_idx:start_idx+n_per_client]
  return indexes_per_client


def noniid_partition(y_train):
  """
    sort the data by digit label, divide it into 200 shards of size 300, and assign each of 100 clients 2 shards.
  """
  n_shards = 200
  n_per_shard = 300

  indexes_per_client = {}
  indexes = y_train.argsort()

  indexes_shard = np.arange(0, n_shards)
  random.shuffle(indexes_shard)

  for i in range(N_CLIENTS):
    start_idx_shard_1 = indexes_shard[i*2]*n_per_shard
    start_idx_shard_2 = indexes_shard[i*2+1]*n_per_shard
    indexes_per_client[i] = np.concatenate((indexes[start_idx_shard_1:start_idx_shard_1+n_per_shard],
                                            indexes[start_idx_shard_2:start_idx_shard_2+n_per_shard]))
    
  return indexes_per_client

def create_batch(indexes_client, X_train, y_train, B):
    x = []
    y = []    
    for i in indexes_client:
      x.append(X_train[i])
      y.append(y_train[i])

    dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))
    return dataset.shuffle(len(y)).batch(len(y_train) if B=='all' else B)


def prepare_data():
  # load mnist data
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

  # prepare iid & noniid data
  indexes_per_client = None
  if DATA=='iid':
   indexes_per_client = iid_partition(y_train)
  elif DATA=='noniid':
    indexes_per_client = noniid_partition(y_train)
  else:
    print('DATA {} is not defined'.format(DATA))

  # normalize
  X_train = X_train.astype("float32")/255
  X_test = X_test.astype("float32")/255
  # expand dim
  X_train = np.expand_dims(X_train, -1)
  X_test = np.expand_dims(X_test, -1)
  # convert to binary class matrix
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)

  print("x_train shape:", X_train.shape)
  print(X_train.shape[0], "train samples")
  print(X_test.shape[0], "test samples")

  client_dataset_batched = {}
  for i, indexes in tqdm(indexes_per_client.items(), desc="Prepare client's data"):
    client_dataset_batched[i] = create_batch(indexes, X_train, y_train, B)

  train_batched = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(len(y_train)) # for testing on train set
  test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
  return client_dataset_batched, train_batched, test_batched

def initialize_model():
  #initialize global model
  model = None
  if MODEL=='mlp':
    model = MLP()
  elif MODEL=='cnn':
    model = CNN()
  else:
    print('model {} is not defined'.format(MODEL))
    
  return model


def train(model, client_dataset_batched, train_batched, test_batched):
  global_model = model.build((28,28,1))
  print(global_model.summary())
  initial_weights = global_model.get_weights()

  client_ids = [i for i in range(N_CLIENTS)]
  loss='categorical_crossentropy'
  metrics = ['accuracy']
  cce = tf.keras.losses.CategoricalCrossentropy()

  result_per_lr = {}
  start = time.time()
  for lr in LRS:
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    optimizer = SGD(lr=lr) 
    global_model.set_weights(initial_weights)
    print('\nlearning rate: {}'.format(lr))
    for r in range(C_ROUNDS):
      train_loss = 0
      train_acc = 0
      test_loss = 0
      test_acc = 0

      global_weights = global_model.get_weights()

      #sampling client
      m = max(int(C*N_CLIENTS), 1)
      selected_clients = random.sample(client_ids, m)

      client_models = {} # to prevent crashed due to not enough RAM 

      # client update
      for i in selected_clients:
        client_models[i] = model.build((28,28,1))
        client_models[i].compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=metrics)
        
        client_models[i].set_weights(global_weights)
        client_models[i].fit(client_dataset_batched[i], epochs=E, verbose=0)

      # averaging
      avg_weights = list()
      for j in range(len(global_weights)):
          weights = [client_models[k].get_weights()[j] for k in selected_clients]
          layer_mean = tf.math.reduce_mean(weights, axis=0)
          avg_weights.append(layer_mean)
      global_model.set_weights(avg_weights)

      # test global model on full training set
      for (X,y) in train_batched:
        preds = global_model.predict(X)
        train_loss = cce(y, preds)
        train_acc = accuracy_score(tf.argmax(preds, axis=1), tf.argmax(y, axis=1))
        train_losses.append(train_loss.numpy())
        train_accs.append(train_acc)

      # test global model on testing set
      for(X, y) in test_batched:
        preds = global_model.predict(X)
        test_loss = cce(y, preds)
        test_acc = accuracy_score(tf.argmax(preds, axis=1), tf.argmax(y, axis=1))
        test_losses.append(test_loss.numpy())
        test_accs.append(test_acc)

      elapsed = (time.time() - start)
      
      print('comm_round: {}/{} | test_acc: {:.3%} | test_loss: {} | train_acc: {:.3%} | train_loss: {} | elapsed: {}'.format(r+1, C_ROUNDS, test_acc, test_loss, train_acc, train_loss, timedelta(seconds=elapsed)))

    result_per_lr[lr] = {
                          'train_accs' : train_accs,
                          'test_accs' : test_accs,
                          'train_losses' : train_losses,
                          'test_losses' : test_losses
                        }
  return global_model, result_per_lr



def get_plotted_metrics(result_per_lr):
  plotted_train_accs= []
  plotted_test_accs = []
  plotted_train_losses = []
  plotted_test_losses = []
  for c in range(C_ROUNDS):
    best_train_acc = 0
    best_test_acc = 0
    best_train_loss = math.inf
    best_test_loss = math.inf
    for lr in result_per_lr.keys():
      best_train_acc = max(best_train_acc, result_per_lr[lr]['train_accs'][c])
      best_test_acc = max(best_test_acc, result_per_lr[lr]['test_accs'][c])
      best_train_loss = min(best_train_loss, result_per_lr[lr]['train_losses'][c])
      best_test_loss = min(best_test_loss, result_per_lr[lr]['test_losses'][c])

    if c == 0:
      plotted_train_accs.append(best_train_acc)
      plotted_test_accs.append(best_test_acc)
      plotted_train_losses.append(best_train_loss)
      plotted_test_losses.append(best_test_loss)
    else:
      if plotted_train_accs[-1] > best_train_acc:
        plotted_train_accs.append(plotted_train_accs[-1])
      else:
        plotted_train_accs.append(best_train_acc)

      if plotted_test_accs[-1] > best_test_acc:
        plotted_test_accs.append(plotted_test_accs[-1])
      else:
        plotted_test_accs.append(best_test_acc)

      if plotted_train_losses[-1] < best_train_loss:
        plotted_train_losses.append(plotted_train_losses[-1])
      else:
        plotted_train_losses.append(best_train_loss)

      if plotted_test_losses[-1] < best_test_loss:
        plotted_test_losses.append(plotted_test_losses[-1])
      else:
        plotted_test_losses.append(best_test_loss)

  return plotted_train_accs, plotted_test_accs, plotted_train_losses, plotted_test_losses


def get_comm_rounds_at_acc(plotted_test_accs, test_acc_target):
  n_round_at_target = None
  for i,acc in enumerate(plotted_test_accs):
    if acc>=test_acc_target:
      n_round_at_target = i+1
      break;
  return n_round_at_target

def get_loss_at_round(plotted_test_losses, n_round_at_target):
  loss_at_target = None
  for i,loss in enumerate(plotted_test_losses):
    if n_round_at_target and i==n_round_at_target-1:
      loss_at_target = loss
      break;
  return loss_at_target

def generate_acc_plot(plotted_train_accs, plotted_test_accs, test_acc_target):
  fig, ax = plt.subplots()
  ax.plot(range(1, len(plotted_train_accs)+1), plotted_train_accs, label='train')

  ax.plot(range(1,len(plotted_test_accs)+1), plotted_test_accs, label='test')
  ax.set_xticks(np.arange(0, len(plotted_test_accs)+1, 100))
  ax.axhline(y=test_acc_target, color='grey', linestyle='-', linewidth=0.5)
  ax.set_ylabel('accuracy')

  ax.set_xlabel('communication rounds')
  ax.set_title('B={}, C={}, E={}, Model={}, Data={}'.format(B, C, E, MODEL, DATA))
  ax.legend()

  ax2 = ax.twinx()
  ax2.set_ylim(ax.get_ylim())
  ax2.set_yticks([test_acc_target])

  # plt.show()
  plt.savefig('train_test_acc_{}_{}_{}_{}_{}.png'.format(B, C, E, MODEL, DATA))

def generate_loss_plot(plotted_train_losses, plotted_test_losses, loss_at_target):
  fig, ax = plt.subplots()
  ax.plot(range(1, len(plotted_train_losses)+1), plotted_train_losses, label='train')
  ax.plot(range(1,len(plotted_test_losses)+1), plotted_test_losses, label='test')
  ax.set_xticks(np.arange(0, len(plotted_test_losses)+1, 100))
  if loss_at_target:
    ax.axhline(loss_at_target, color='grey', linestyle='-', linewidth=0.5)
  ax.set_ylabel('loss')
  ax.set_xlabel('communication rounds')
  ax.set_title('B={}, C={}, E={}, Model={}, Data={}'.format(B, C, E, MODEL, DATA))
  ax.legend()

  ax2 = ax.twinx()
  ax2.set_ylim(ax.get_ylim())
  if loss_at_target:
    ax2.set_yticks([loss_at_target])

  # plt.show()
  plt.savefig('train_test_loss_{}_{}_{}_{}_{}.png'.format(B, C, E, MODEL, DATA))
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="mlp or cnn", choices=['mlp', 'cnn'], required=True)
  parser.add_argument("--data_dist", help="iid or noniid", choices=['iid', 'noniid'], required=True)
  parser.add_argument("--n_clients", help="number of clients", type=int, default=100, required=True)
  parser.add_argument('--c', help="client fraction", type=float, choices=[Range(0.0, 1.0)], default=0, required=True)
  parser.add_argument('--e', help="client epoch", type=int, default=1, required=True)
  parser.add_argument('--b', help="batch size. input -1 to use a single batch", type=int, default=-1, required=True)
  parser.add_argument('--c_rounds', help='communication rounds', type=int, default=200, required=True)
  parser.add_argument('--lr', nargs='+', help='learning rate. separate by space for multiple learning rates. eg. --lr 0.1 0.001', default=0.1, required=True)
  parser.add_argument('--target_acc', help="target test accuracy", type=float, choices=[Range(0.0, 1.0)], default=0.93, required=True)


  args = parser.parse_args() 


  MODEL = args.model
  DATA = args.data_dist
  N_CLIENTS = args.n_clients
  C = args.c
  E = args.e
  B = args.b if args.b!=-1 else 'all'
  C_ROUNDS = args.c_rounds
  LRS = args.lr # learning rate
  LRS = [float(l) for l in LRS]
  TARGET_ACC = args.target_acc

  random_seed()

  # prepare data
  client_dataset_batched, train_batched, test_batched = prepare_data()
  print('data is prepared')

  # initialize model
  model = initialize_model()
  print("model {} is initialized".format(MODEL.upper()))


  # training
  print("Training....")
  global_model, train_result = train(model,
    client_dataset_batched,
    train_batched,
    test_batched)

  #save the train result (use for plotting)
  with open('train_result_{}_{}_{}_{}_{}.pickle'.format(B,C,E,MODEL,DATA), 'wb') as handle:
    pickle.dump(train_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('train result is saved')

  # save the trained model
  global_model.save('model_{}_{}_{}_{}_{}'.format(B,C,E,MODEL,DATA))
  print('model is saved')

  # generate and save plot
  plotted_train_accs, plotted_test_accs, plotted_train_losses, plotted_test_losses = get_plotted_metrics(train_result)
  
  generate_acc_plot(plotted_train_accs, plotted_test_accs, TARGET_ACC)
  print('train-test accuracy plot is saved')

  n_round_at_target = get_comm_rounds_at_acc(plotted_test_accs, TARGET_ACC)
  loss_at_target = get_loss_at_round(plotted_test_losses, n_round_at_target)
  
  generate_loss_plot(plotted_train_losses, plotted_test_losses, loss_at_target)
  print('train-test loss plot is saved')

  print()
  print()
  print('Model: {}'.format(MODEL))
  print('Data: {}'.format(DATA))
  print('C (Client fraction): {}'.format(C))
  print('B (Batch size): {}'.format(B))
  print('E (Epoch): {}'.format(E))
  print('Number of clients: {}'.format(N_CLIENTS))
  print('Communication rounds: {}'.format(C_ROUNDS))
  print('Learning rates: {}'.format(LRS))
  print('Target acc: {}'.format(TARGET_ACC))
  print("Number of rounds to achieve target test-accuracy: {}".format(n_round_at_target))
  print("Loss at target test-accuracy: {}".format(loss_at_target))
  acc_plot_file = 'train_test_acc_{}_{}_{}_{}_{}.png'.format(B, C, E, MODEL, DATA)
  print("Acc plot: {}".format(acc_plot_file))
  loss_plot_file = 'train_test_loss_{}_{}_{}_{}_{}.png'.format(B, C, E, MODEL, DATA)
  print("Loss plot: {}".format(loss_plot_file))  


