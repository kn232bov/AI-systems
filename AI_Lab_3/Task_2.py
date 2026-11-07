import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

# Визначення універсумів
temp_domain = np.linspace(0, 100, 101)
pres_domain = np.linspace(0, 10, 101)
angle_domain = np.linspace(-90, 90, 181)

# Вхідні та вихідні нечіткі змінні
temperature = ctrl.Antecedent(temp_domain, 'temperature')
pressure = ctrl.Antecedent(pres_domain, 'pressure')
hot_valve = ctrl.Consequent(angle_domain, 'hot_valve')
cold_valve = ctrl.Consequent(angle_domain, 'cold_valve')

# Терми для температури
temperature['cold'] = fuzzy.trimf(temp_domain, [0, 0, 25])
temperature['cool'] = fuzzy.trimf(temp_domain, [10, 30, 50])
temperature['warm'] = fuzzy.trimf(temp_domain, [40, 50, 60])
temperature['not_very_hot'] = fuzzy.trimf(temp_domain, [55, 70, 85])
temperature['hot'] = fuzzy.trimf(temp_domain, [75, 100, 100])

# Терми для напору
pressure['weak'] = fuzzy.trimf(pres_domain, [0, 0, 4])
pressure['not_very_strong'] = fuzzy.trimf(pres_domain, [2, 5, 8])
pressure['strong'] = fuzzy.trimf(pres_domain, [6, 10, 10])

# Терми для кутів повороту кранів
for valve in [hot_valve, cold_valve]:
    valve['large_left']   = fuzzy.trimf(angle_domain, [-90, -60, -30])
    valve['medium_left']  = fuzzy.trimf(angle_domain, [-50, -30, -10])
    valve['small_left']   = fuzzy.trimf(angle_domain, [-30, -15, 0])
    valve['zero']         = fuzzy.trimf(angle_domain, [-10, 0, 10])
    valve['small_right']  = fuzzy.trimf(angle_domain, [0, 15, 30])
    valve['medium_right'] = fuzzy.trimf(angle_domain, [10, 30, 50])
    valve['large_right']  = fuzzy.trimf(angle_domain, [30, 60, 90])

# Створення 11 правил згідно з текстом завдання
water_rules = [
    ctrl.Rule(temperature['hot'] & pressure['strong'], (hot_valve['medium_left'], cold_valve['medium_right'])),
    ctrl.Rule(temperature['hot'] & pressure['not_very_strong'], cold_valve['medium_right']),
    ctrl.Rule(temperature['not_very_hot'] & pressure['strong'], hot_valve['small_left']),
    ctrl.Rule(temperature['not_very_hot'] & pressure['weak'], (hot_valve['small_right'], cold_valve['small_right'])),
    ctrl.Rule(temperature['warm'] & pressure['not_very_strong'], (hot_valve['zero'], cold_valve['zero'])),
    ctrl.Rule(temperature['cool'] & pressure['strong'], (hot_valve['medium_right'], cold_valve['medium_left'])),
    ctrl.Rule(temperature['cool'] & pressure['not_very_strong'], (hot_valve['medium_right'], cold_valve['small_left'])),
    ctrl.Rule(temperature['cold'] & pressure['weak'], hot_valve['large_right']),
    ctrl.Rule(temperature['cold'] & pressure['strong'], (hot_valve['medium_left'], cold_valve['medium_right'])),
    ctrl.Rule(temperature['warm'] & pressure['strong'], (hot_valve['small_left'], cold_valve['small_left'])),
    ctrl.Rule(temperature['warm'] & pressure['weak'], (hot_valve['small_right'], cold_valve['small_right']))
]

water_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(water_rules))

test_water = [
    (1, 90.0, 9.0),  (2, 90.0, 5.0),  (3, 70.0, 9.0),
    (4, 70.0, 1.5),  (5, 50.0, 5.0),  (6, 30.0, 9.0),
    (7, 30.0, 5.0),  (8, 5.0, 1.5),   (9, 5.0, 9.0),
    (10, 50.0, 9.0), (11, 50.0, 1.5)
]

print("=== WATER TAP TESTING ===")
print(f"{'№ Rules':<7} | {'Temp (°C)':<9} | {'Pressure':<8} | {'Hot Valve':<10} | {'Cold Valve':<10}")
print("-" * 55)

for rule_no, t, p in test_water:
    water_sim.input['temperature'] = t
    water_sim.input['pressure'] = p
    water_sim.compute()
    h_out = water_sim.output.get('hot_valve', 0.0)
    c_out = water_sim.output.get('cold_valve', 0.0)
    print(f"Rule {rule_no:<2} | {t:<9.1f} | {p:<8.1f} | {h_out:<10.2f} | {c_out:<10.2f}")