import wandb

wandb.login()
wandb.init(project="hubmap-unext", entity="mglee_")

for i in range(10):
    wandb.log({'test':i})