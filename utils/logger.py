import os
import sys
import time

def get_run_logdir(logdir:str):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    dst_logdir = os.path.join(logdir, run_id)
    os.makedirs(dst_logdir, exist_ok=True)
    return dst_logdir