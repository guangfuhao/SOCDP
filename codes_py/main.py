from model_train import *
from dataset import *
from utils import *
import argparse


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--d', '--data', metavar='DIR', default="./data/mnist",help='path to dataset (default: mnist)')
    parser.add_argument('--n', '--run-times-per-class',  metavar='TIMES', default=100, type=int,help='run times per class (default: 100)')
    parser.add_argument('--ms', '--middle-size', default=8, type=int,help='model middle size (default: 8)')
    parser.add_argument('--cnr', metavar='class_num_runs', default=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),type=list_of_ints, help='class num runs')
    parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-sp', '--save-path', default="./output/",metavar='N', help='model save path (default: "./output/")')
    return parser


parser = get_args_parser()

'''
python main.py --d "../data/mnist" --n 100 --batch-size 64 --ms 16 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "./new/"
python main.py --d "../data/mnist" --n 100 --batch-size 64 --ms 32 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "./new/"
python main.py --d "../data/mnist" --n 100 --batch-size 64 --ms 64 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "./new/"
python main.py --d "../data/mnist" --n 100 --batch-size 64 --ms 128 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "./new/"
python main.py --d "../data/mnist" --n 100 --batch-size 64 --ms 256 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "./new/"
'''
if __name__ == "__main__":
    args = parser.parse_args()
    print("data DIR:{}".format(args.d))
    print("run times per class:{}".format(args.n))
    print("mini-batch size:{}".format(args.batch_size))
    print("model save path:{}".format(args.save_path))

    middle_size = args.ms
    class_num_runs = [x * 10 for x in args.cnr]
    print("model middle size:{}".format(args.ms))
    print("class num runs:{}".format(class_num_runs))
    model_config = [8, 16, middle_size, middle_size * middle_size // 4]
    print("model_config:", model_config)

    for class_num_run in class_num_runs:
        print("class num run:{}".format(class_num_run))
        train_data, test_data = load_processed_dataset(data_dir=args.d, batch_size=args.batch_size, display=False,
                                                       process=True,
                                                       class_num=class_num_run)
        for i in range(1, args.n + 1):
            for model_name in MODEL_NAME:
                train(train_data, test_data, train_idx=i, model_name=model_name, class_num=class_num_run,
                      lambda_sparse=0.01, out_dir=args.save_path, model_config=model_config)
