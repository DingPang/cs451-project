import numpy as np
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors
from sklearn.linear_model import SGDClassifier
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
scores = {}
acc_mean = []
acc_std = []

# Which subset of data will potentially really matter.
for step in steps:
    # n_samples = int((train_percent / 100) * N)
    n_samples = step
    # print("{}% == {} samples...".format(train_percent, n_samples))
    print("{} samples...".format(n_samples))
    # label = "{}".format(train_percent, n_samples)
    label = "{}".format(n_samples)

    # So we consider num_trials=100 subsamples, and train a model on each.
    scores[label] = []
    for i in range(num_trials):
        X_sample, y_sample = resample(
            train_X_Feature, train_y_Feature, n_samples=n_samples, replace=False
        )  # type:ignore
        # Note here, I'm using a simple classifier for speed, rather than the best.
        clf = SGDClassifier(random_state=RANDOM_SEED + i)
        clf.fit(X_sample, y_sample)
        # so we get 100 scores per percentage-point.
        scores[label].append(clf.score(vali_X_Feature, vali_y_Feature))
    # We'll first look at a line-plot of the mean:
    acc_mean.append(np.mean(scores[label]))
    acc_std.append(np.std(scores[label]))

# First, try a line plot, with shaded variance regions:
import matplotlib.pyplot as plt

means = np.array(acc_mean)
std = np.array(acc_std)
plt.plot(steps, acc_mean, "o-")
plt.fill_between(steps, means - std, means + std, alpha=0.2)
plt.xlabel("Samples Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, N])
plt.title("Shaded Accuracy Plot")
plt.savefig("graphs/simpleSGD.png")
plt.show()


# Second look at the boxplots in-order: (I like this better, IMO)
simple_boxplot(
    scores,
    "Learning Curve",
    xlabel="Samples Training Data",
    ylabel="Accuracy",
    save="graphs/simpleSGD-boxplot.png",
)





# Experimenting codes:
# print("type_to_label dict: " + str(type_to_label))
# img = cv2.imread("./fruits-360/Test/Apple Braeburn/3_100.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("image",img)
# cv2.waitKey()
