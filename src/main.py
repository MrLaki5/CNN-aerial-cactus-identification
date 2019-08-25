import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import keras.preprocessing.image as image
import numpy as np

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


def create_model():
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
                            activation='relu'
                            # Activation function that puts 0 on all negative values and linear for other
                            )
              )

    # Layers that reduce number of parameters when images are too large.
    model.add(layers.MaxPooling2D(pool_size=(2, 2),  # 2 integers, factors by which to downscale (vertical, horizontal)
                                  padding='same'  # Adds some zeros as padding if image is not right size
                                  )
              )

    # To prevent overfitting dropout layer is add which at same rate randomly sets input to 0
    model.add(layers.Dropout(0.3  # Dropout rate
                             )
              )

    # 2nd part of convolution
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(0.3))

    # 3rd part of convolution
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(0.3))

    # 4th part of convolution
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(0.3))

    # Flattens input to 1D
    model.add(layers.Flatten())
    # Creates hidden, highly connected, layer
    model.add(layers.Dense(256,  # Number of neurons in hidden layer
                           activation='relu'
                           # Activation function that puts 0 on all negative values and linear for other
                           )
              )
    # Creates output layer
    model.add(layers.Dense(1,  # Output layer has 1 neuron
                           activation='sigmoid'  # Activation function that returns array of probabilities, is or is not
                           )
              )

    # Compiling the model

    # Optimizers are algorithms that provide how will weights be changed in training of model.
    # Stochastic Gradient Descent - algorithm that is used for finding minimum of function with derivative of function
    # and looking at its slope. (Closer to minimum, slope is closer to 0). Every weight has same
    # learning rate (value that is multiplied in formula for weight changing when minimum of
    # function has not been found).
    optimizer = Adam(lr=0.0022, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer,  # Optimizer
                  loss='binary_crossentropy',  # Loss function
                  metrics=['accuracy']  # Metrics to be used in calculating how good model is trained
                  )

    return model


def train_model():
    # Create image preprocessor, that can do augmentation on images (this can help with accuracy of end nn)
    image_gen = ImageDataGenerator(rescale=1. / 255,  # Scale values to borders of 1 (normalize)
                                   )

    # Load images which names are in csv file from specific directory
    train_generator = image_gen.flow_from_dataframe(whole_set_csv[:15000],  # Which images to be loaded
                                                    directory=train_data_dir,  # Directory where images are
                                                    x_col='id',  # Name of the image that is in csv file
                                                    y_col='has_cactus',  # Output value that is in csv file
                                                    target_size=(64, 64),  # Size to which images will be resized
                                                    class_mode='binary',  # 1d numpy array of binary labels (is/not cactus)
                                                    batch_size=BATCH_SIZE,  # Size of batches of data
                                                    )

    validation_generator = image_gen.flow_from_dataframe(whole_set_csv[15000:],
                                                         directory=train_data_dir,
                                                         x_col='id',
                                                         y_col='has_cactus',
                                                         target_size=(64, 64),
                                                         class_mode='binary',
                                                         batch_size=BATCH_SIZE,
                                                         )

    # Create model
    model = create_model()

    model.summary()

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


def test_model():
    # Check if model exists
    if not os.path.exists('./model_checkpoint.hdf5'):
        print("Model is not created!")
        return
    # Load model
    model = create_model()
    model.load_weights("model_checkpoint.hdf5")

    # Load test images
    test_file = []
    for i in os.listdir("../data_set/test/"):
        if i.endswith(".jpg"):
            test_file.append(i)
    test_image = []
    for f_name in test_file:
        img = image.load_img('../data_set/test/'+f_name, target_size=(64, 64, 3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    test_img_array = np.array(test_image)

    # Inference test images
    output = model.predict_classes(test_img_array)

    # Save results to file
    submission_save = pd.DataFrame()
    submission_save['id'] = test_file
    submission_save['has_cactus'] = output
    submission_save.to_csv('../data_set/submission.csv', header=True, index=False)


def main():
    work_flag = True
    while work_flag:
        print("----------------")
        print("Menu:")
        print("1. Create and train model")
        print("2. Inference test")
        print("3. Exit")
        user_action = input("Action: ")
        if user_action == "1":
            train_model()
        elif user_action == "2":
            test_model()
        elif user_action == "3":
            work_flag = False


main()
