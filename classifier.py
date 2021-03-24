from typing import Counter, Dict
import warnings
import numpy as np
from tqdm import tqdm
import os
import cv2
import random

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
      img = cv2.imread(img_Path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      test_X.append(np.array(img))
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
      img = cv2.imread(img_Path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      tv_X.append(np.array(img))
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

# Experimenting codes:
# print("type_to_label dict: " + str(type_to_label))
# img = cv2.imread("./fruits-360/Test/Apple Braeburn/3_100.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("image",img)
# cv2.waitKey()
