import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "../data_set/train"
# Load csv file
whole_set_csv = pd.read_csv("../data_set/train.csv")
# y_col values must be strings in binary class mode
whole_set_csv.has_cactus = whole_set_csv.has_cactus.astype(str)

# Size of batches of images
BATCH_SIZE = 128

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
