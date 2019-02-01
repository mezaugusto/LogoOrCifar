from glob import glob
from os import makedirs, getcwd
from os.path import join, isdir, exists
from numpy import array, zeros, ones, float32, concatenate
from numpy.random import seed, shuffle, get_state, set_state
from pickle import load, dump
from sys import stdout
from .constants import *
import urllib
import tarfile
from tqdm import tqdm
import zipfile

seed(SEED)
img_size = MODEL_IMAGE_SIZE
num_channels = IMAGE_CHANNELS
num_classes = CIFAR_CLASSES


def _cut_array(arr, lenght):
    """
    Truncates an array shuffling its elements
    :param arr: a list of elements
    :param lenght: desired array length
    :return: list of size length
    """
    if 0 < lenght < len(arr):
        shuffle(arr)
        return arr[:lenght]
    return arr


def _unpickle(filename, data_path):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    :param filename: String for the file
    :param data_path: String for the folder
    :return: the unpickled data
    """

    file_path = join(data_path, filename)

    print(loading_message(file_path))

    with open(file_path, mode='rb') as file:
        data = load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format to match LLD
    where the pixels are floats between 0.0 and 1.0 and the arrengment is
    [#, H, W, C]
    :param raw: A CIFAR-10 format array
    :return: A 4-dim array with shape: [image_number, height, width, channel]
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename, data_path):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    :param filename: cifar-10 file
    :param data_path: cifar-10 dataset path
    :return: images ready for the model
    """

    # Load the pickled data-file.
    try:
        data = _unpickle(filename, data_path)

        # Get the raw images.
        raw_images = data[b'data']

        # Convert the images.
        images = _convert_images(raw_images)

        return images
    except FileNotFoundError as e:
        print(e)
        error_message(CIFAR_NOT_PRESENT + data_path)
        exit()

def _download_and_extract_cifar(data_path):
    print(DOWNLOADING_CIFAR)
    tmp_file = DATASET_PATH+'CIFAR.zip'
    if not exists(tmp_file):
        with TqdmUpTo(desc='Downloading') as t:
            urllib.request.urlretrieve(CIFAR_DATASET_URL, filename=tmp_file, reporthook=t.update_to)
    with tarfile.open(name=tmp_file) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc='Extracting'):
            tar.extract(member=member, path=DATASET_PATH)

def load_cifar_data(cifar_len, data_path=CIFAR_DATASET_PATH):
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    :param cifar_len: size for the cifar dataset
    :param data_path: path to the cifar dataset
    :return: the complete cifar dataset ready for training
    """
    if not exists(data_path):
        _download_and_extract_cifar(data_path)

    _num_files_train = CIFAR_FILES

    # Total number of images in the training-set.
    # This is used to pre-allocate arrays for efficiency.
    _num_images_train = _num_files_train * CIFAR_IMAGES_PER_FILE

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = zeros(shape=[_num_images_train, img_size, img_size, num_channels],
                   dtype=float)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images from the data-file.
        images_batch = _load_data(CIFAR_FILE_PREFIX + str(i + 1), data_path)

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return _cut_array(images, cifar_len)

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def _download_and_extract_lld(data_path):
    print(DOWNLOADING_LLD)
    tmp_file = DATASET_PATH+'LLD.zip'
    if not exists(tmp_file):
        with TqdmUpTo(desc='Downloading') as t:
            urllib.request.urlretrieve(LLD_DATASET_URL, filename=tmp_file, reporthook=t.update_to)
    with zipfile.ZipFile(tmp_file,"r") as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()), desc='Extracting'):
            zip_file.extract(member=file, path=DATASET_PATH)

def _extract_individual_images(icons):
    create_dirs(INDIVIDUAL_IMAGES_PATH)
    step = 100 / len(icons)
    for i, img in enumerate(icons):
        with open('datasets/individuals/img_' + str(i), 'wb') as file:
            dump(img, file)
        stdout.write('\rCreating miniatures {:2.2f}%'.format(step * (i + 1)))


def load_lld_data(lld_len, data_path=LLD_DATASET_PATH, single_file=None):
    """
    Load the entire LLD and create training dataset
    :param lld_len: length for the new LLD
    :param data_path: folder for the LLD
    :param single_file: Int, if only one file of LLD is needed
    :return: the sub-dataset of LLD
    """
    if not exists(LLD_DATASET_PATH):
        _download_and_extract_lld(data_path)

    pattern = LLD_FILE_PATTERN
    num_files = LLD_FILES
    files = glob(join(data_path, pattern))
    if len(files) <= 0:
        print(LLD_NOT_PRESENT.format(data_path))

    files.sort()
    if single_file is None:
        with open(files[0], 'rb') as f:
            print(loading_message(files[0]))
            icons = load(f, encoding='bytes')
        if len(files) > 1:
            for file in files[1:num_files]:
                print(loading_message(file))
                with open(file, 'rb') as f:
                    icons_loaded = load(f, encoding='bytes')
                icons = concatenate((icons, icons_loaded))
    else:
        with open(files[single_file % len(files)], 'rb') as f:
            icons = load(f)
    return _cut_array(icons, lld_len)


def create_dataset(cifar, lld):
    """
    Create full dataset, shuffling while preserving labels
    :param cifar: the cifar sub-dataset
    :param lld: the lld sub-dataset
    :return: the training dataset images and labels
    """
    print(CREATING_DATASET)
    dataset_size = len(cifar) + len(lld)
    step = 100 / dataset_size
    images = zeros(shape=[dataset_size, img_size, img_size, num_channels],
                   dtype=float32)
    i, j = 0, 0
    for i, image in enumerate(cifar):
        images[i] = image
        stdout.write('\r{} {:2.2f}%'.format(loading_message('CIFAR'),
                                            step * (i + 1)))
    for j, image in enumerate(lld):
        images[i + j] = image
        stdout.write('\r{} {:2.2f}%'.format(loading_message('LLD'),
                                            step * (i + j + 1)))

    labels = zeros(shape=[dataset_size], dtype=int)
    labels[len(cifar):dataset_size] = ones(shape=[len(lld)], dtype=int)

    del cifar, lld
    print(CREATION_SUCCESFULL)
    rng_state = get_state()
    shuffle(images)
    set_state(rng_state)
    shuffle(labels)
    return images, labels


def load_and_create_dataset(cifar_len, lld_len):
    """
    Load datasets and create full datasets
    :param cifar_len: size for the cifar dataset
    :param lld_len: size for the lld dataset
    :return: the full dataset
    """
    lld = load_lld_data(lld_len)
    print(LLD_LOADED)
    cifar = load_cifar_data(cifar_len)
    print(CIFAR_LOADED)
    print('LLD Shape:', lld.shape, 'CIFAR Shape:', cifar.shape)
    return create_dataset(cifar, lld)


def create_dirs(rel_path):
    """
    Create any necessary parent directory
    :param rel_path: the relative path needed
    :return: the complete path from the cwd
    """
    save_dir = join(getcwd(), rel_path)
    if not isdir(save_dir):
        makedirs(save_dir)
    return save_dir
