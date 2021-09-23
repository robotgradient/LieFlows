import numpy as np
s = np.linspace(0.0, 2 * np.pi, 100)[None, :]
t = np.linspace(-1.0, 1.0, 50)[:, None]


x = (1 + 0.5 * t * np.cos(0.5 * s)) * np.cos(s)
y = (1 + 0.5 * t * np.cos(0.5 * s)) * np.sin(s)
z = 0.5 * t * np.sin(0.5 * s)
P = np.stack([x, y, z], axis=-1)


N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
N /= np.sqrt(np.sum(N ** 2, axis=-1))[:, :, None]


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

ax.scatter(P[:,:,0], P[:,:,1], P[:,:,2],c = 'r')


plt.show()
