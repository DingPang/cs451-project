import tensorflow as tf
import os
from tensorflow.keras.applications import vgg19, VGG19
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors

# This is a class extened on tf.keras.models.Model
class myVGG(tf.keras.models.Model):
  def __init__(self, layer):
      super(myVGG, self).__init__()
      # get vgg19
      vgg = VGG19(include_top=False, weights="imagenet")

      features = vgg.get_layer(layer).output

      # Define this model with our selected inputs/outputs
      self.vgg = tf.keras.Model([vgg.input], [features])
      # Not trainable, just for features extraction
      self.vgg.trainable = False

  def call(self, inputs):
      features = self.vgg(inputs)
      return features
