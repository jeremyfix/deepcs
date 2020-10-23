# coding: utf-8

import os

def generate_unique_logpath(logdir, raw_run_name):
    '''
    Generate a unique directory name

    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name

    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    '''
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1
