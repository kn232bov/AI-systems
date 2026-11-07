import numpy as np
import matplotlib.pyplot as plt

n = 15
x1 = np.linspace(0, 4, n)
x2 = np.linspace(0, 4, n)
X1, X2 = np.meshgrid(x1, x2)
Y = (X1**2 - 8) * np.cos(X2)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 1. Використовуємо колірну палітру 'jet' з чорною сіткою
surf = ax.plot_surface(X1, X2, Y, cmap='jet', edgecolor='k', linewidth=0.5)

# 2. Повертаємо камеру в точно такий самий ракурс, як у MATLAB
ax.view_init(elev=30, azim=-120)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Target')

plt.tight_layout()
plt.show()