import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

train_data_dir = "../data_set/train"
# Load csv file
whole_set_csv = pd.read_csv("../data_set/train.csv")
# y_col values must be strings in binary class mode
whole_set_csv.has_cactus = whole_set_csv.has_cactus.astype(str)

# Size of batches of images
BATCH_SIZE = 128

# File path for model checkpoints
MODEL_PATH = "model_checkpoint.hdf5"

# Number of max epochs in model training
MAX_EPO = 30

# Create image preprocessor, that can do augmentation on images (this can help with accuracy of end nn)
image_gen = ImageDataGenerator(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               zca_epsilon=1e-06,
                               rotation_range=60,  # Degree range for random rotations
                               width_shift_range=0.15,  # Shift fraction of total width
                               height_shift_range=0.15,  # Shift fraction of total height
                               brightness_range=None,
                               shear_range=0.0,
                               zoom_range=0.0,
                               channel_shift_range=0.0,
                               fill_mode='nearest',
                               cval=0.0,
                               horizontal_flip=True,  # Flip images horizontally
                               vertical_flip=False,  # Don't flip images vertically, coz it cactus can't grow that way.
                               rescale=1. / 255,  # Scale values to borders of 1 (normalize)
                               preprocessing_function=None,
                               data_format=None,
                               validation_split=0.2,  # Split set for validation
                               dtype=None
                               )

# Load images which names are in csv file from specific directory
train_generator = image_gen.flow_from_dataframe(whole_set_csv,  # Which images to be loaded
                                                directory=train_data_dir,  # Directory where images are
                                                x_col='id',  # Name of the image that is in csv file
                                                y_col='has_cactus',  # Output value that is in csv file
                                                target_size=(64, 64),  # Size to which images will be resized
                                                color_mode='rgb',  # Color channels to which images will be converted
                                                classes=None,
                                                class_mode='binary',  # 1d numpy array of binary labels (is/not cactus)
                                                batch_size=BATCH_SIZE,  # Size of batches of data
                                                shuffle=True,  # Should shuffle data
                                                seed=None,
                                                save_to_dir=None,
                                                save_prefix='',
                                                save_format='png',
                                                subset='training',  # 'training' or 'validation' if validation_split set
                                                interpolation='nearest',  # Technique to be used if resizing is needed
                                                drop_duplicates=True  # Drop duplicates based on filename
                                                )

validation_generator = image_gen.flow_from_dataframe(whole_set_csv,
                                                     directory=train_data_dir,
                                                     x_col='id',
                                                     y_col='has_cactus',
                                                     target_size=(64, 64),
                                                     color_mode='rgb',
                                                     classes=None,
                                                     class_mode='binary',
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     seed=None,
                                                     save_to_dir=None,
                                                     save_prefix='',
                                                     save_format='png',
                                                     subset='validation',
                                                     interpolation='nearest',
                                                     drop_duplicates=True
                                                     )

# Building a model
model = Sequential()

# For CNN networks firstly add convolution layer, layer which takes input features. But instead of taking pixel by
# pixel it takes matrixes of pixels that are near each other and applies to them some filter that can provide some
# attributes to show clearly (etc. edges, blurriness...)


# 1st part of convolution
# Conv2D is layer that takes 2D arrays (images)
model.add(layers.Conv2D(filters=32,  # Dimensionality of the output space, at start it was 3 (color formats)
                        kernel_size=(3, 3),  # Dimensionality of conv window that is being looked and filtered
                        input_shape=(64, 64, 3),  # Size and color format of input
                        activation='relu'  # Activation function that puts 0 on all negative values and linear for other
                        )
          )

# Layers that reduce number of parameters when images are too large.
model.add(layers.MaxPooling2D(pool_size=(2, 2),  # 2 integers, factors by which to downscale (vertical, horizontal)
                              padding='same'  # Adds some zeros as padding if image is not right size
                              )
          )

# To prevent overfitting dropout layer is add which at same rate randomly sets input to 0
model.add(layers.Dropout(0.25  # Dropout rate
                         )
          )


# 2nd part of convolution
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(layers.Dropout(0.25))

# 3rd part of convolution
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(layers.Dropout(0.25))

# 4th part of convolution
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(layers.Dropout(0.25))

# Flattens input to 1D
model.add(layers.Flatten())
# Creates hidden, highly connected, layer
model.add(layers.Dense(128,  # Number of neurons in hidden layer
                       activation='relu'  # Activation function that puts 0 on all negative values and linear for other
                       )
          )
# Creates output layer
model.add(layers.Dense(1,  # Output layer has 1 neuron
                       activation='softmax'  # Activation function that returns array of probabilities, is or is not
                       )
          )


# Compiling the model

# Optimizers are algorithms that provide how will weights be changed in training of model.
# Stochastic Gradient Descent - algorithm that is used for finding minimum of function with derivative of function
# and looking at its slope. (Closer to minimum, slope is closer to 0). Every weight has same learning rate (value that
# is multiplied in formula for weight changing when minimum of function has not been found).
optimizer = SGD(lr=0.001,  # Learning rate
                momentum=0.0,
                decay=0.0,
                nesterov=False
                )


model.compile(optimizer=optimizer,  # Optimizer
              loss='mean_squared_error',  # Loss function
              metrics=['accuracy']  # Metrics to be used in calculating how good model is trained
              )


# Model will be trained in epochs, model with best accuracy or reduced loss will be used for test.
# That's the reason why checkpoints will be used.
checkpoint = ModelCheckpoint(MODEL_PATH,
                             monitor='val_acc',  # Quantity to monitor.
                             verbose=1,
                             save_best_only=True,  # Best model according to the quantity monitored will be saved.
                             mode='max'  # How to look at quantity to monitor, which is the best.
                             )
callbacks_list = [checkpoint]


# Train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,  # One step is one batch processed from train generator
    epochs=MAX_EPO,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    validation_steps=30.  # One step is one batch processed from validation generator
)

# Plot accuracy of validation per epochs
val_acc = history.history['val_acc']
epochs = range(1, len(val_acc) + 1)
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.legend()
plt.show()
