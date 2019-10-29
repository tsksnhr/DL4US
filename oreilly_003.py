import numpy as np
import oreilly_002

def XOR(X):

    s1 = oreilly_002.B_nand(X)
    s2 = oreilly_002.B_or(X)

    s_ = np.array([s1, s2])

    out = oreilly_002.B_and(s_)

    print(out)

print("XOR")
XOR(np.array([0, 0]))
XOR(np.array([0, 1]))
XOR(np.array([1, 0]))
XOR(np.array([1, 1]))
