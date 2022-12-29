
import argparse
import os
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import re

def get_plotted_metrics(result_per_lr):
  plotted_train_accs= []
  plotted_test_accs = []
  plotted_train_losses = []
  plotted_test_losses = []

  comm_rounds = None
  for lr in result_per_lr.keys():
    comm_rounds = len(result_per_lr[lr]['train_accs'])
    break
  
  for c in range(comm_rounds):
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



def generate_plot(model, data, result_files):
  result_files.sort()
  colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:yellow']
  fig, ax = plt.subplots(figsize=(14,7))
  for i,file in enumerate(result_files):
    with open(file, 'rb') as handle:
      train_result = pickle.load(handle)
    plotted_train_accs, plotted_test_accs, plotted_train_losses, plotted_test_losses = get_plotted_metrics(train_result)

    batch = re.search('(?<=train_result_)(.*?)(?=_)', file).groups()[0]
    epoch = re.search('(?:.*?_){4}(.*?)(?=_)', file).groups()[0]
    batch = '∞' if batch=='all' else batch

    color = colors[int(i/2)%3]
    linestyle = '-' if i%2==0 else '--'
    ax.plot(range(1, len(plotted_test_accs)+1), plotted_test_accs, label='B={}  E={}'.format(batch, epoch), color=color, linestyle=linestyle)
  ax.set_xticks(np.arange(0, len(plotted_test_accs)+1, 100))
  ax.axhline(y=0.99, color='grey', linestyle='-', linewidth=0.5)
  ax.set_ylabel('test accuracy')

  ax.set_xlabel('communication rounds')
  ax.set_title('MNIST {} {}'.format(MODEL.upper(), DATA.upper()))
  ax.legend()

  # ax.set_ylim((0.9, 1.0005))

  ax2 = ax.twinx()
  ax2.set_ylim(ax.get_ylim())
  ax2.set_yticks([0.99])

  plt.savefig('multi_cases_acc_{}_{}.png'.format(MODEL, DATA))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="mlp or cnn", choices=['mlp', 'cnn'], required=True)
  parser.add_argument("--data_dist", help="iid or noniid", choices=['iid', 'noniid'], required=True)
  parser.add_argument('--result_files', nargs='+', help='training result files. separate by space for multiple files. eg. --result_files file1.pickle file2.pickle', required=True)

  args = parser.parse_args() 

  MODEL = args.model
  DATA = args.data_dist
  RESULT_FILES = args.result_files

  generate_plot(MODEL, DATA, RESULT_FILES)