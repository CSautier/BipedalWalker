import torch


class Parameters:
    def __init__(self):
        self.ACTOR_COEFF = 0.02
        # self.ENTROPY_COEFF = 0.000005
        self.LOSS_CLIPPING = 0.15
        self.GAMMA = 0.95
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.OBS_SPACE, self.ACTION_SPACE = 24, 4
        self.BATCH_SIZE = 32
        self.EPOCH_STEPS = 10
        self.BURN_IN = 10
        self.MAXLEN = 10000
        self.STD_MODIFICATION_RATE = 0.05


parameters = Parameters()
