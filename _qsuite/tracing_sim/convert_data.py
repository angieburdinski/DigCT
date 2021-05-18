import numpy as np

x = np.array([
                [   [1,2,3],
                    [4,5,6] ],
                [   [1,5,3],
                    [4,5,6] ]
                    ])
print(x.shape)
print(np.mean(x,axis=0).shape)
print(np.mean(x,axis=2).shape)
try:
    print(y)
    print(x)
except Exception:
    pass

print("Exception ignored")
try:
    print(y)
    print(x)
except NameError:
    pass
print(x)
