# Logo Or CIFAR
Logo Or CIFAR is a CLI tool that trains a model to classify between [LLD](https://data.vision.ee.ethz.ch/sagea/lld/) (Large Logo Dataset) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). 


# How to run the project
> Note: This project was tested on `Python 3.6` and thus it is recommended

1. Preferably in a clean virtual environment install the required packages with `pip install -r requirements.txt`
2. To use the program with the defaults simply run in the project root `python app.py`

# Additional options

Run `python app.py --help` to see the configuration options including:
  - Specify the size for the total training dataset and how many images per dataset it must contain
  - Clean the dataset folder before running