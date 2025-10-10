import numpy as np
import os

control = np.load("/home/jyc3887/differentiable-thermomechanical-AM/stress_melt_liq+500/bfgs/control_0018.npy", allow_pickle=True)

print(np.mean(control[:100]))