import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

# Домени змінних
temp_dom = np.linspace(10, 35, 101)
dt_dom = np.linspace(-3, 3, 101)
reg_dom = np.linspace(-90, 90, 181)

temp = ctrl.Antecedent(temp_dom, 'temp')
dt = ctrl.Antecedent(dt_dom, 'dt')
regulator = ctrl.Consequent(reg_dom, 'regulator')

temp['very_cold'] = fuzzy.trimf(temp_dom, [10, 10, 16])
temp['cold']      = fuzzy.trimf(temp_dom, [14, 18, 21])
temp['normal']    = fuzzy.trimf(temp_dom, [19, 22, 25])
temp['warm']      = fuzzy.trimf(temp_dom, [23, 27, 30])
temp['very_warm'] = fuzzy.trimf(temp_dom, [28, 35, 35])

dt['negative'] = fuzzy.trimf(dt_dom, [-3, -3, 0])
dt['zero']     = fuzzy.trimf(dt_dom, [-0.5, 0, 0.5])
dt['positive'] = fuzzy.trimf(dt_dom, [0, 3, 3])

regulator['large_left']  = fuzzy.trimf(reg_dom, [-90, -60, -30])
regulator['small_left']  = fuzzy.trimf(reg_dom, [-40, -20, 0])
regulator['off']         = fuzzy.trimf(reg_dom, [-10, 0, 10])
regulator['small_right'] = fuzzy.trimf(reg_dom, [0, 20, 40])
regulator['large_right'] = fuzzy.trimf(reg_dom, [30, 60, 90])

ac_rules = [
    ctrl.Rule(temp['very_warm'] & dt['positive'], regulator['large_left']),
    ctrl.Rule(temp['very_warm'] & dt['negative'], regulator['small_left']),
    ctrl.Rule(temp['warm'] & dt['positive'], regulator['large_left']),
    ctrl.Rule(temp['warm'] & dt['negative'], regulator['off']),
    ctrl.Rule(temp['very_cold'] & dt['negative'], regulator['large_right']),
    ctrl.Rule(temp['very_cold'] & dt['positive'], regulator['small_right']),
    ctrl.Rule(temp['cold'] & dt['negative'], regulator['large_left']),
    ctrl.Rule(temp['cold'] & dt['positive'], regulator['off']),
    ctrl.Rule(temp['very_warm'] & dt['zero'], regulator['large_left']),
    ctrl.Rule(temp['warm'] & dt['zero'], regulator['small_left']),
    ctrl.Rule(temp['very_cold'] & dt['zero'], regulator['large_right']),
    ctrl.Rule(temp['cold'] & dt['zero'], regulator['small_right']),
    ctrl.Rule(temp['normal'] & dt['positive'], regulator['small_left']),
    ctrl.Rule(temp['normal'] & dt['negative'], regulator['small_right']),
    ctrl.Rule(temp['normal'] & dt['zero'], regulator['off'])
]

ac_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(ac_rules))

test_ac = [
    (1, 32.0, 2.0),   (2, 32.0, -2.0),  (3, 27.0, 2.0),
    (4, 27.0, -2.0),  (5, 11.0, -2.0),  (6, 11.0, 2.0),
    (7, 17.5, -2.0),  (8, 17.5, 2.0),   (9, 32.0, 0.0),
    (10, 27.0, 0.0),  (11, 11.0, 0.0),  (12, 17.5, 0.0),
    (13, 22.0, 2.0),  (14, 22.0, -2.0), (15, 22.0, 0.0)
]

print("\n=== AIR CONDITIONER TESTING ===")
print(f"{'№ Rules':<7} | {'Temp (°C)':<9} | {'dt (°C/min)':<11} | {'Regulator Angle':<15}")
print("-" * 50)

for rule_no, t, d in test_ac:
    ac_sim.input['temp'] = t
    ac_sim.input['dt'] = d
    ac_sim.compute()
    r_out = ac_sim.output.get('regulator', 0.0)
    print(f"Rule {rule_no:<2} | {t:<9.1f} | {d:<11.1f} | {r_out:<15.2f}")