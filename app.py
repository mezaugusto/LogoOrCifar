from logocifar import LogoOrCifar

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser("Configure LogoOrCifar")
    p.add_argument("--train-size", default=150000, type=int,
                   action="store", dest="train_sz",
                   help="Train sample size [150000]")
    p.add_argument("--cifar-size", default=50000, type=int,
                   action="store", dest="train_sz",
                   help="Cifar sample size 0 for all [0]")
    p.add_argument("--verbose", default=False,
                   action="store", dest="verbose",
                   help="Verbose output [False]")
    p.add_argument("--force", default=False,
                   action="store", dest="forceRecal",
                   help="Force vector recalculation [False]")

    opts = p.parse_args()

LogoOrCifar(
    train_sz=opts.train_sz,
    cifar_len=1,
    lld_len=1
)
