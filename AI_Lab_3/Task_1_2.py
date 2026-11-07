import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Створення нечітких змінних
x1_domain = np.linspace(0, 4, 101)
x2_domain = np.linspace(0, 4, 101)
y_domain = np.linspace(-10, 10, 101)

x1 = ctrl.Antecedent(x1_domain, 'x1')
x2 = ctrl.Antecedent(x2_domain, 'x2')
y = ctrl.Consequent(y_domain, 'y')

# 2. Функції належності для x1 (3 трикутні терми з параметрами MATLAB)
x1['L'] = fuzzy.trimf(x1.universe, [-1.6, 0, 1.6])
x1['A'] = fuzzy.trimf(x1.universe, [0, 2, 4])
x1['H'] = fuzzy.trimf(x1.universe, [2.4, 4, 5.6])

# 3. Функції належності для x2 (5 гауссівських термів з параметрами MATLAB)
x2['L']  = fuzzy.gaussmf(x2.universe, 0.0, 0.4247)
x2['LA'] = fuzzy.gaussmf(x2.universe, 1.0, 0.4247)
x2['A']  = fuzzy.gaussmf(x2.universe, 2.0, 0.4247)
x2['HA'] = fuzzy.gaussmf(x2.universe, 3.0, 0.4247)
x2['H']  = fuzzy.gaussmf(x2.universe, 4.0, 0.4247)

# 4. Функції належності для y (5 трикутних термів з параметрами MATLAB)
y['L']  = fuzzy.trimf(y.universe, [-15, -10, -5])
y['LA'] = fuzzy.trimf(y.universe, [-10, -5, 0])
y['A']  = fuzzy.trimf(y.universe, [-5, 0, 5])
y['HA'] = fuzzy.trimf(y.universe, [0, 5, 10])
y['H']  = fuzzy.trimf(y.universe, [5, 10, 15])

# 5. Формування правил нечіткої бази знань
rule1 = ctrl.Rule(x1['L'] & x2['L'], y['L'])
rule2 = ctrl.Rule(x1['L'] & x2['H'], y['A'])
rule3 = ctrl.Rule(x1['L'] & x2['HA'], y['H'])
rule4 = ctrl.Rule(x1['H'] & x2['L'], y['HA'])
rule5 = ctrl.Rule(x1['H'] & x2['H'], y['L'])
rule6 = ctrl.Rule(x1['A'] & x2['A'], y['A'])
rule7 = ctrl.Rule(x1['A'] & x2['HA'], y['HA'])
rule8 = ctrl.Rule(x1['L'] & x2['LA'], y['LA'])
rule9 = ctrl.Rule(x1['A'] & x2['HA'], y['A'])

# 6. Побудова та симуляція нечіткої системи
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
sim = ctrl.ControlSystemSimulation(system)

# Обчислення виходу для контрольної точки x1=2, x2=2
sim.input['x1'] = 2.0
sim.input['x2'] = 2.0
sim.compute()
print(f"Розраховане значення y при (x1=2, x2=2): {sim.output['y']:.4f}")

# 7. Побудова нечіткої поверхні "входи-вихід" (Surface Viewer)
n_grid = 15  # 15х15 сітка, як у MATLAB (X grids = 15, Y grids = 15)
x1_g = np.linspace(0, 4, n_grid)
x2_g = np.linspace(0, 4, n_grid)
X1_grid, X2_grid = np.meshgrid(x1_g, x2_g)
Y_fis = np.zeros_like(X1_grid)

for i in range(n_grid):
    for j in range(n_grid):
        sim.input['x1'] = X1_grid[i, j]
        sim.input['x2'] = X2_grid[i, j]
        sim.compute()
        Y_fis[i, j] = sim.output['y']

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Оформлення 3D-поверхні під стиль MATLAB Surface Viewer
surf = ax.plot_surface(X1_grid, X2_grid, Y_fis, cmap='jet', edgecolor='k', linewidth=0.4)

# Поворот камери під ракурс MATLAB
ax.view_init(elev=30, azim=-120)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y (FIS Output)')
ax.set_title('Surface Viewer: first2 (Mamdani FIS)')

plt.tight_layout()
plt.show()