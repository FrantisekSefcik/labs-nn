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
    x = keras.layers.Dense(1, activation='sigmoid', name='dense_b')(x)
    return keras.Model(y, x)


def initialize_slicenet_model_2(IMG_HEIGHT, IMG_WIDTH):
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