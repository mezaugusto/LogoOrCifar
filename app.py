from logocifar import LogoOrCifar
from logocifar.constants import *
import argparse


def clean_datasets():
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser("Configure LogoOrCifar")
    p.add_argument("--train-size", default=150000, type=int,
                   action="store", dest="train_sz",
                   help="Train sample size [150000]")
    p.add_argument("--cifar-size", default=0, type=int,
                   action="store", dest="cifar_sz",
                   help="Cifar sample size, 0 for all [0]")
    p.add_argument("--lld-size", default=50000, type=int,
                   action="store", dest="lld_sz",
                   help="LLD sample size, 0 for all [50000]")
    p.add_argument("--clean", default=False, type=bool,
                   action="store", dest="clean",
                   help="Clean all datasets")

    opts = p.parse_args()

    try:
        if opts.clean:
            clean_datasets()
        model = LogoOrCifar(
            train_sz=opts.train_sz,
            cifar_len=opts.cifar_sz,
            lld_len=opts.lld_sz
        )
        model.define_model()
        model.fit_to_data()
    except ValueError as e:
        error_message(LLD_NOT_PRESENT)
        print(e)
