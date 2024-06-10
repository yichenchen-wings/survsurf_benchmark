import numpy as np


class IPCW:
    def __init__(self, time, S_censor):
        self.time = time
        self.S_censor = S_censor
    def __call__(self,t):
        S_censor = self.S_censor
        percentile  = np.percentile(S_censor, 10)
        S_censor[S_censor <= percentile] = percentile
        S = np.interp(t, self.time, S_censor)
        return 1/S