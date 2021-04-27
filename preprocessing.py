import numpy as np
from tqdm import tqdm
import os
import random
import math
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors
import joblib


type_to_label = {}
IMG_SIZE = 100

# Test Data
TEST_DIR_PATH = "./fruits-360/Test"
test_X=[]
test_y=[]

def make_test_data(DIR):
  label = 0
  for subDIR in tqdm(os.listdir(DIR)):
    # ignore Hidden dirs
    if (subDIR.startswith('.')): continue
    subDIR_PATH = os.path.join(DIR, subDIR)
    type_to_label[subDIR] = label
    for img in os.listdir(subDIR_PATH):
      img_Path = os.path.join(subDIR_PATH, img)
      #####
      img = image.load_img(img_Path, target_size=(IMG_SIZE, IMG_SIZE))
      img_data = image.img_to_array(img)
      img_data = np.expand_dims(img_data, axis=0)
      img_data = preprocess_input(img_data)
      # img = cv2.imread(img_Path, cv2.IMREAD_COLOR)
      # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      test_X.append(np.array(img_data))
      test_y.append(str(label))
    label +=1


# Trainning Data
TRAIN_DIR_PATH = "./fruits-360/Training"
tv_X=[]
tv_y=[]

def make_tv_data(DIR):
  for subDIR in tqdm(os.listdir(DIR)):
    # ignore Hidden dirs
    if (subDIR.startswith('.')): continue
    subDIR_PATH = os.path.join(DIR, subDIR)
    label = type_to_label[subDIR]
    for img in os.listdir(subDIR_PATH):
      img_Path = os.path.join(subDIR_PATH, img)
      #####
      img = image.load_img(img_Path, target_size=(IMG_SIZE, IMG_SIZE))
      img_data = image.img_to_array(img)
      img_data = np.expand_dims(img_data, axis=0)
      img_data = preprocess_input(img_data)
      # img = cv2.imread(img_Path, cv2.IMREAD_COLOR)
      # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      tv_X.append(np.array(img_data))
      tv_y.append(str(label))

train_X = []
train_y = []
vali_X = []
vali_y = []

def seperate_tv(tv_X, tv_y):
  tv_X_copy = tv_X.copy()
  tv_y_copy = tv_y.copy()
  # shuffle
  temp = list(zip(tv_X_copy, tv_y_copy))
  random.shuffle(temp)
  resX, resy = zip(*temp)
  # splits
  totalLen = len(resy)
  sep = int(4*totalLen/5)
  global train_X, train_y, vali_X, vali_y
  train_X = list(resX)[:sep].copy()
  train_y = list(resy)[:sep].copy()
  vali_X = list(resX)[sep:].copy()
  vali_y = list(resy)[sep:].copy()


print("Getting Test Data")
make_test_data(TEST_DIR_PATH)
print("Test Data Created!")
print("Num of Images in Test Data: " + str(len(test_y)))


print("Getting Train/Validation Data")
make_tv_data(TRAIN_DIR_PATH)
seperate_tv(tv_X, tv_y)
print("Train Data Created!")
print("Num of Images in Train Data: " + str(len(train_X)))
print("Validation Data Created!")
print("Num of Images in Validation Data: " + str(len(vali_X)))

base_model = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg", weights='imagenet')
# base_model.summary()
base_model.trainable = False

train_X_Feature = []
train_y_Feature = []
vali_y_Feature = []
vali_X_Feature = []
# My dataset is too big for my computer to handle
shrink = 0.1
for i in tqdm(range(math.floor(shrink*len(train_X)))):
  x_Feature = base_model.predict(train_X[i])
  x_Feature = np.array(x_Feature)
  train_X_Feature.append(x_Feature.flatten())
  train_y_Feature.append(train_y[i])


for i in tqdm(range(math.floor(shrink*len(vali_X)))):
  x_Feature = base_model.predict(vali_X[i])
  x_Feature = np.array(x_Feature)
  vali_X_Feature.append(x_Feature.flatten())
  vali_y_Feature.append(vali_y[i])

data_dir = "./featuredata"
try:
  os.mkdir(data_dir)
except FileExistsError:
  pass

train_X_Feature_Path = os.path.join(data_dir, "train_X_Feature")
train_y_Feature_Path = os.path.join(data_dir, "train_y_Feature")
vali_X_Feature_Path = os.path.join(data_dir, "vali_X_Feature")
vali_y_Feature_Path = os.path.join(data_dir, "vali_y_Feature")

for data, path in zip(
  [train_X_Feature, train_y_Feature, vali_X_Feature, vali_y_Feature],
  [train_X_Feature_Path, train_y_Feature_Path, vali_X_Feature_Path, vali_y_Feature_Path]):
  joblib.dump(data, path)
