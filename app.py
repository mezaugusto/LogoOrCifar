from logocifar import LogoOrCifar

import argparse
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

    opts = p.parse_args()

    LogoOrCifar(
        train_sz=opts.train_sz,
        cifar_len=opts.cifar_sz,
        lld_len=opts.lld_sz
    )
