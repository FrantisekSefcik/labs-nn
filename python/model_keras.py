import tensorflow as tf
from tensorflow import keras


def initialize_brats_model(IMG_HEIGHT, IMG_WIDTH):
    y = keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                           name='input_layer')
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                            name='conv_a')(y)
    x = keras.layers.MaxPooling2D(name='max_a')(x)
    x = keras.layers.Dropout(0.2, name='drop_a')(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                            name='conv_b')(x)
    x = keras.layers.MaxPooling2D(name='max_b')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                            name='conv_c')(x)
    x = keras.layers.MaxPooling2D(name='max_c')(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu',
                            name='conv_d')(x)
    x = keras.layers.MaxPooling2D(name='max_d')(x)
    x = keras.layers.Dropout(0.2, name='drop_c')(x)
    x = keras.layers.Flatten(name='flat_a')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_a')(x)
    x = keras.layers.Dense(2, name='dense_b')(x)
    return keras.Model(y, x)


def initialize_slicenet_model(IMG_HEIGHT, IMG_WIDTH):
    y = keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='input_layer')
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='conv_a')(y)
    x = keras.layers.MaxPooling2D(name='max_a', pool_size=[3,3])(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='conv_b')(x)
    x = keras.layers.MaxPooling2D(name='max_b', pool_size=[3,3])(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv_c')(x)
    x = keras.layers.MaxPooling2D(name='max_c', pool_size=[3,3])(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv_d')(x)
    x = keras.layers.MaxPooling2D(name='max_d')(x)
    x = keras.layers.Dropout(0.2, name='drop_c')(x)
    x = keras.layers.Flatten(name='flat_a')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_a')(x)
    x = keras.layers.Dense(2, name='dense_b')(x)
    return keras.Model(y, x)

# Create an oveerride model to classify pictures
class BratsModel(tf.keras.Model):
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, **kwargs):
        super(BratsModel, self).__init__(**kwargs)
        self.input_layer = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='input_layer')
        self.conv_1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='conv_1')
        self.max_1 = tf.keras.layers.MaxPooling2D(name='max_1')
        self.drop_1 = tf.keras.layers.Dropout(0.2, name='drop_1')
        self.conv_2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='conv_2')
        self.max_2 = tf.keras.layers.MaxPooling2D(name='max_2')
        self.conv_3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv_3')
        self.max_3 = tf.keras.layers.MaxPooling2D(name='max_3')
        self.conv_4 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv_4')
        self.max_4 = tf.keras.layers.MaxPooling2D(name='max_4')
        self.drop_2 = tf.keras.layers.Dropout(0.2, name='drop_2')
        self.flat = tf.keras.layers.Flatten(name='flat')
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(2, name='dense_2')

    def call(self, x):
        x = self.input_layer(x)
        x = self.conv_1(x)
        x = self.max_1(x)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.max_2(x)
        x = self.conv_3(x)
        x = self.max_3(x)
        x = self.conv_4(x)
        x = self.max_4(x)
        x = self.drop_2(x)
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x