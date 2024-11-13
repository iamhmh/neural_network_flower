import numpy as np 

x_entry =  np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float)
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # 1 = red, 0 = blue

x_entry = x_entry/np.amax(x_entry, axis=0) # scaling the data to 0-1

print(x_entry)