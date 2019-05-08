import argparse
import sys

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', type=str, help='path to the snapshot file (a pickled dict)')
    parser.add_argument(
        '--p_num',
        default=None,
        type=str,
        help='snapshot number within pickle')
    parser.add_argument('--total', type=int, default=0)
    parser.add_argument(
        '--max_path_length',
        type=int,
        default=1000,
        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    parser.add_argument(
        '--save',
        default=None,
        help='video will be saved  to given file instead of being displayed')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if args.p_num is not None:
            try:
                data2 = data[args.p_num]
            except KeyError:
                msg = 'No such key "%s"; valid keys are "%s"' \
                      % (args.p_num, ', '.join(data.keys()))
                print(msg, file=sys.stderr)
                sys.exit(1)
        else:
            data2 = data
        policy = data2['policy']
        env = data2['env']
        while True:
            if args.total > 0:
                for i in range(args.total):
                    env = data[str(i)]['env']
                    policy = data[str(i)]['policy']

                    path = rollout(
                        env,
                        policy,
                        max_path_length=args.max_path_length,
                        animated=True,
                        speedup=args.speedup,
                        animated_save_path=args.save)
            else:
                path = rollout(
                    env,
                    policy,
                    max_path_length=args.max_path_length,
                    animated=True,
                    speedup=args.speedup,
                    animated_save_path=args.save)
            if not query_yes_no('Continue simulation?'):
                break
