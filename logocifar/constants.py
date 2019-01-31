"""
    LITERAL CONSTANTS
"""
SEED = 42
# Width and height of each image.
MODEL_IMAGE_SIZE = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
IMAGE_CHANNELS = 3

# Number of classes for CIFAR
CIFAR_CLASSES = 10

# Number of images for each batch-file in CIFAR
CIFAR_IMAGES_PER_FILE = 10000

LOADING_MESSAGE = 'Loading '

CREATING_DATASET = 'Creating the combined dataset'
CREATION_SUCCESFULL = '\nDataset Created, Shuffling (this can take a while)...'

LLD_LOADED = 'LLD Data loaded'
CIFAR_LOADED = 'CIFAR Data loaded'

DATASET_EXISTS = 'Dataset already exists, loading...'
SAVING_DATASET = 'Saving dataset in'

MODEL_TRAINED = 'Model: {} has been trained, exiting program'
SAVED_TRAINED = 'Saved trained model at'


def loading_message(extra=''):
    return LOADING_MESSAGE + extra


# PATHNAME CONSTANTS
DATASET_PATH = 'datasets/'
CIFAR_DATASET_PATH = '{}cifar-10-batches-py/'.format(DATASET_PATH)
LLD_DATASET_PATH = '{}LLD-icon'.format(DATASET_PATH)
PROCESSED_DATASET_PATH = '{}processed/'.format(DATASET_PATH)

CIFAR_FILE_PREFIX = 'data_batch_'
LLD_FILE_PATTERN = 'LLD-icon_data_*.pkl'

LLD_FILES = 6
CIFAR_FILES = 5

MDL_FILE_EXT = '_dataset.pkl'
MODEL_PATH = 'saved_models'


def format_model_name(cifar_len, lld_len):
    return 'cifar_{0}_lld_{1}'.format(cifar_len, lld_len)


def trained_model(flat, conv):
    return '_flat_{0}_conv_{1}.h5'.format(str(flat), str(conv))


# ERROR CONSTANTS
GENERAL_ERROR = 'ERROR: Model could not be created'
LLD_NOT_PRESENT = 'LLD Dataset not present in {}, downloading...'
CIFAR_NOT_PRESENT = 'CIFAR Dataset not present in '
NOT_ENOUGH_MEMORY = 'Not enough memory to save dataset'


def error_message(description):
    print(GENERAL_ERROR)
    print('\t -', description)

# DATASET URL's
LLD_DATASET_URL = 'https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-icon_PKL.zip'
CIFAR_DATASET_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DOWNLOADING_CIFAR = 'CIFAR not found, downloading and extracting'
DOWNLOADING_LLD = 'LLD not found, downloading and extracting'