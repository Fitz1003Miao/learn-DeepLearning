import tensorflow as tf
import argparse
import AlexNet
import importlib

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--train", help = "Path to train file")
    parse.add_argument("-v", "--val", help = "Path to val file")
    parse.add_argument("-s", "--setting", help = "setting to import")
    args = parse.parse_args()
    print(args)

    train_file = args.train
    val_file = args.val
    
    setting = importlib.import_module(args.setting)

if __name__ == "__main__":
    main()