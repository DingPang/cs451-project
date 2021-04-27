import numpy as np
from tensorflow.python.keras.engine.training import Model
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from shared import simple_boxplot
from sklearn.utils import resample
import joblib

data_dir = "./featuredata"

train_X_Feature_Path = os.path.join(data_dir, "train_X_Feature")
train_y_Feature_Path = os.path.join(data_dir, "train_y_Feature")
vali_X_Feature_Path = os.path.join(data_dir, "vali_X_Feature")
vali_y_Feature_Path = os.path.join(data_dir, "vali_y_Feature")

train_X_Feature = joblib.load(train_X_Feature_Path,  mmap_mode='r')
train_y_Feature = joblib.load(train_y_Feature_Path,  mmap_mode='r')
vali_X_Feature = joblib.load(vali_X_Feature_Path,  mmap_mode='r')
vali_y_Feature = joblib.load(vali_y_Feature_Path,  mmap_mode='r')

RANDOM_SEED = 12345678
N = len(train_y_Feature)
oneStep = 50
numOfSteps = 10
steps = list(range(oneStep, oneStep*numOfSteps, oneStep))
steps.append(N)
num_trials = len(steps)
model_scores = {}
# acc_mean = []
# acc_std = []

for name, base_model in zip(
  ["sgd", "perceptron"],
  [SGDClassifier, Perceptron]):
  model_scores[name] = {}
  model_scores[name]['acc_mean'] = []
  model_scores[name]['acc_std'] = []
  for step in steps:
      n_samples = step
      print("{}-{} samples...".format(name, n_samples))
      label = "{}".format(n_samples)

      model_scores[name][label] = []

      for i in range(num_trials):
          X_sample, y_sample = resample(
              train_X_Feature, train_y_Feature, n_samples=n_samples, replace=False
          )
          model = base_model(random_state=RANDOM_SEED + i)
          model.fit(X_sample, y_sample)
          model_scores[name][label].append(model.score(vali_X_Feature, vali_y_Feature))
      # We'll first look at a line-plot of the mean:
      model_scores[name]['acc_mean'].append(np.mean(model_scores[name][label]))
      model_scores[name]['acc_std'].append(np.std(model_scores[name][label]))

import matplotlib.pyplot as plt

for name, value in model_scores.items():
  means = np.array(model_scores[name]['acc_mean'])
  std = np.array(model_scores[name]['acc_std'])
  plt.plot(steps, means, "o-", label="{}".format(name))
  plt.fill_between(steps, means - std, means + std, alpha=0.2)

plt.xlabel("Samples Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, N])
plt.legend()
plt.title("Shaded Accuracy Plot")
plt.savefig("graphs/simpleSGD.png")
plt.show()



# simple_boxplot(
#     scores,
#     "Learning Curve",
#     xlabel="Samples Training Data",
#     ylabel="Accuracy",
#     save="graphs/simpleSGD-boxplot.png",
# )



# TODO:
# Maybe try resNet50
# Not just linear model?
