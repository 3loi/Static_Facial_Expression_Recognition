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
#MODEL_PATH = "output/vggface_4emo_73.model"
MODEL_CONTINUE_PATH = "continue.model"
MODEL_PATH = "continue.model"
MODEL_PATH = "output/affectnet_4emo2.model"


MODEL_PATH_TRAIN = "/media/a/storage/4emo_train_vgg.hdf5"
MODEL_PATH_VAL = "/media/a/storage/4emo_val_vgg.hdf5"

MODEL_PATH_TRAIN = "/media/a/storage/4emo_train.hdf5"
MODEL_PATH_VAL = "/media/a/storage/4emo_val.hdf5"

MODEL_PATH_TRAIN = "/home/a/Desktop/train4emo.hdf5"
MODEL_PATH_VAL = "/home/a/Desktop/test4emo.hdf5"
BATCH_SIZE = 64
EPOCHS = 50

IMAGE_WIDTH = 227 
IMAGE_HEIGHT = 227
# define the path to the dataset mean
DATASET_MEAN = "output/affectnet_downsample_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
