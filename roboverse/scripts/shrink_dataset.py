import argparse
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-n', '--num-traj', type=int, required=True)
    args = parser.parse_args()

    print('loading data..')
    data = np.load(args.input, allow_pickle=True)
    output = data[np.random.choice(data.shape[0], args.num_traj, replace=False)]
    print('saving..')
    np.save(args.output, output)
