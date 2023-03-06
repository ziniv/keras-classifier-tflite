import os
cur_dir = os.getcwd()
import sys
import subprocess
import glob
import argparse
import zipfile

def get_TinyImageNet(extract, output):
    if extract:
        output_file = os.path.join(output, "tiny-imagenet-200.zip")
        if not os.path.exists(output_file):
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            subprocess.run(["wget", url, "-O", output_file])
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(path=output)
        

def get_Mnist(extract, output):
    import tarfile
    if extract:
        output_file = os.path.join(output, "mnist_png.tar.gz")
        if not os.path.exists(output_file):
            url = "https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
            subprocess.run(["wget", url, "-O", output_file])
        with tarfile.open(output_file, "r:gz") as tar_ref:
            tar_ref.extractall(path=output)


def get_Cifar10(extract, output):
    import tarfile
    if extract:
        output_file = os.path.join(output, "cifar-10-python.tar.gz")
        if not os.path.exists(output_file):
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            subprocess.run(["wget", url, "-O", output_file])
        with tarfile.open(output_file, "r:gz") as tar_ref:
            tar_ref.extractall(path=output)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="To prepare datasets")
    parser.add_argument('--name', default='TinyImageNet', type=str)
    parser.add_argument('--extract', default=False, type=bool)
    parser.add_argument('--output', default=cur_dir, type=str)
    args = parser.parse_args()
    if args.name.lower() == 'tinyimagenet':
        get_TinyImageNet(args.extract, args.output)
    elif args.name.lower() == 'mnist':
        get_Mnist(args.extract, args.output)
    elif args.name.lower() == 'cifar':
        get_Cifar10(args.extract, args.output)


