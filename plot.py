import os

import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# a = torch.load("models/highway-fast-v0_PPO_0.log")
# b = torch.load("models/highway-fast-v0_ShieldPPO_0.log")
# #c = torch.load("models/highway-fast-v0_RuleBasedShieldPPO_0.log")
# c = torch.load("models/highway-fast-no-normalization-v0_RuleBasedShieldPPO_0.log")

"""
a = torch.load("models/highway-fast-no-normalization-v0_PPO_0_20220127-094915.log")
b = torch.load("models/highway-fast-no-normalization-v0_ShieldPPO_0_20220127-094935.log")
c = torch.load("models/highway-fast-no-normalization-v0_RuleBasedShieldPPO_0_20220127-094943.log")
"""

a = torch.load("models\ShieldPPO_1_20230801-111422\stats.log")

def cost_to_cost_rate(array):
    global acc, i, tmp
    acc = 0
    for i in range(len(array[2])):
        tmp = array[2][i]
        array[2][i] += acc
        array[2][i] /= (i+1)
        acc += tmp


cost_to_cost_rate(a)


print(acc)
#cost_to_cost_rate(b)
#cost_to_cost_rate(c)


# def smooth(x):
#     n = x.shape[0]
#     b = np.zeros((n,))
#     for i in range(n):
#         b[i] = x[max(0, i-50):min(n, i+50)].mean()
#     return b

def smooth(x):
    return x

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(a[0], smooth(np.array(a[1])), label="PPO", color='r')
#axs[0].plot(b[0], smooth(np.array(b[1])), label="ShieldPPO", color='b')
#axs[0].plot(c[0], smooth(np.array(c[1])), label="RuleBasedShieldPPO", color='g')
#axs[0].set_yscale('log')
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Return")
#axs[1].plot(a[0], smooth(np.array(a[2])), label="PPO", color='r')
#axs[1].plot(b[0], smooth(np.array(b[2])), label="ShieldPPO", color='b')
#axs[1].plot(c[0], smooth(np.array(c[2])), label="RuleBasedShieldPPO", color='g')
#axs[1].set_yscale('log')
#axs[1].set_xlabel("Steps")
#axs[1].set_ylabel("Mistake Rate")


axs[0].legend()
#axs[1].legend()

plt.tight_layout()
plt.show()
#plt.savefig("result3.png", dpi=100)

