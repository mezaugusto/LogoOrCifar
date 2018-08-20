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


def loading_message(extra=''):
    return LOADING_MESSAGE + extra


"""
    PATHNAME CONSTANTS
"""
CIFAR_DATASET_PATH = 'datasets/cifar-10/'
LLD_DATASET_PATH = 'datasets/LLD'

CIFAR_FILE_PREFIX = 'data_batch_'
LLD_FILE_PATTERN = 'LLD-icon_data_*.pkl'

LLD_FILES = 6
CIFAR_FILES = 5

"""
    ERROR CONSTANTS
"""
GENERAL_ERROR = 'ERROR: Model could not be created'
LLD_NOT_PRESENT = 'LLD Dataset not present in '
CIFAR_NOT_PRESENT = 'CIFAR Dataset not present in '


def error_message(description):
    print(GENERAL_ERROR)
    print('\t -', description)
