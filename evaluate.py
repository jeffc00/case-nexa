import argparse
import logging
from os import listdir, path

import numpy as np

from file_evaluation import evaluate_file


logger = logging.getLogger()


def main():
    args = get_arguments()
    set_logger(args.debug)
    errors = np.array([])
    for file_path in get_files_in_folder(args.folder_name):
        file_errors = evaluate_file(file_path)
        errors = np.append(errors, file_errors)
    score = get_rms(errors)
    logger.info('Your score was %f', score)


def get_rms(values):
    """
    Returns the root-mean-square of a numpy array
    >>> get_rms(np.array([0.0]))
    0.0
    >>> get_rms(np.array([3.0, -4.0, -2.0, 14.0]))
    7.5
    """
    return np.sqrt(np.mean(values**2))


def get_files_in_folder(folder_name):
    for file_name in listdir(folder_name):
        file_path = path.join(folder_name, file_name)
        if path.isfile(file_path):
            yield file_path


def set_logger(debug_level):
    if debug_level:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('folder_name', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
