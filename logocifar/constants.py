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

MODEL_TRAINED = 'Model has been trained'
SAVED_TRAINED = 'Saved trained model at'


def loading_message(extra=''):
    return LOADING_MESSAGE + extra


"""
    PATHNAME CONSTANTS
"""
CIFAR_DATASET_PATH = 'datasets/cifar-10/'
LLD_DATASET_PATH = 'datasets/LLD'
PROCESSED_DATASET_PATH = 'datasets/processed/'

CIFAR_FILE_PREFIX = 'data_batch_'
LLD_FILE_PATTERN = 'LLD-icon_data_*.pkl'

LLD_FILES = 6
CIFAR_FILES = 5

MDL_FILE_EXT = '_dataset.pkl'
MODEL_PATH = 'saved_models'


def model_name(cifar_len, lld_len):
    return 'cifar_{0}_lld_{1}'.format(cifar_len, lld_len)


def trained_model(flat, conv):
    return '_flat_{0}_conv_{1}.h5'.format(str(flat), str(conv))


"""
    ERROR CONSTANTS
"""
GENERAL_ERROR = 'ERROR: Model could not be created'
LLD_NOT_PRESENT = 'LLD Dataset not present in '
CIFAR_NOT_PRESENT = 'CIFAR Dataset not present in '
NOT_ENOUGH_MEMORY = 'Not enough memory to save dataset'


def error_message(description):
    print(GENERAL_ERROR)
    print('\t -', description)
