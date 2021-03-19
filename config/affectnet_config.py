# define the paths to the images directory
IMAGES_PATH = "../datasets/affectnet/train"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 4
NUM_VAL_IMAGES = 200 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../train4emo.hdf5"
VAL_HDF5 = "../test4emo.hdf5"
TEST_HDF5 = "../test4emo.hdf5"

# path to the output model file
MODEL_PATH = "output/affectnet_4emo2.model"

BATCH_SIZE = 64

IMAGE_WIDTH = 227 
IMAGE_HEIGHT = 227
# define the path to the dataset mean
DATASET_MEAN = "output/affectnet_downsample_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
