import wandb
import sys, getopt

# wandb.login()
# wandb.init(project="hubmap-unext", entity="mglee_")

argv = sys.argv

opts, etc_args = getopt.getopt(argv[1:],"e:",["env="])

for opt, arg in opts:
    if arg == 'kaggle':
        import math
        print(math.ceil(1.7))