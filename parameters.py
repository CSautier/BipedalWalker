import torch


class Parameters:
    def __init__(self):
        self.ACTOR_COEFF = 0.1
        self.CENTERING_COEFF = 0.00001
        self.LOSS_CLIPPING = 0.15
        self.GAMMA = 0.90
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.OBS_SPACE, self.ACTION_SPACE = 24, 4
        self.BATCH_SIZE = 64
        self.EPOCH_STEPS = 20
        self.MAXLEN = 5000


parameters = Parameters()
