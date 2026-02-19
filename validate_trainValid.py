import numpy as np

train = np.fromfile("data/train.bin", dtype=np.uint16)
val = np.fromfile("data/val.bin", dtype=np.uint16)

print("Train tokens:", len(train))
print("Val tokens:", len(val))
print("First 10 tokens:", train[:10])

