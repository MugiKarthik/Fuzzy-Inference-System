import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
physical_condition = ctrl.Antecedent(np.arange(0, 11, 1), 'Physical Condition')
mental_disposition = ctrl.Antecedent(np.arange(0, 11, 1), 'Mental Disposition')
emotional_resilience = ctrl.Antecedent(np.arange(0, 11, 1), 'Emotional Resilience')
artistic_inclination = ctrl.Antecedent(np.arange(0, 11, 1), 'Artistic Inclination')
overall_capability = ctrl.Consequent(np.arange(0, 101, 1), 'Overall Capability')

# Auto-membership function population
physical_condition.automf(3)
mental_disposition.automf(3)
emotional_resilience.automf(3)
artistic_inclination.automf(3)

# Custom membership functions for overall_capability
overall_capability['very_low'] = fuzz.trimf(overall_capability.universe, [0, 0, 25])
overall_capability['low'] = fuzz.trimf(overall_capability.universe, [0, 25, 50])
overall_capability['medium'] = fuzz.trimf(overall_capability.universe, [25, 50, 75])
overall_capability['high'] = fuzz.trimf(overall_capability.universe, [50, 75, 100])
overall_capability['very_high'] = fuzz.trimf(overall_capability.universe, [75, 100, 100])

# Visualize membership functions (optional)
physical_condition.view()
mental_disposition.view()
emotional_resilience.view()
artistic_inclination.view()
overall_capability.view()


# Define fuzzy rules using existing membership functions
rule1 = ctrl.Rule(physical_condition['good'] & mental_disposition['good'] & emotional_resilience['good'] & artistic_inclination['good'], overall_capability['very_high'])
rule2 = ctrl.Rule(physical_condition['good'] & mental_disposition['good'] & emotional_resilience['average'] & artistic_inclination['average'], overall_capability['high'])
rule3= ctrl.Rule(physical_condition['average'] & mental_disposition['average'] & emotional_resilience['average'] & artistic_inclination['average'], overall_capability['medium'])
rule4= ctrl.Rule(physical_condition['average'] & mental_disposition['average'] & emotional_resilience['poor'] & artistic_inclination['poor'], overall_capability['low'])
rule5 = ctrl.Rule(physical_condition['poor'] & mental_disposition['poor'] & emotional_resilience['poor'] & artistic_inclination['poor'], overall_capability['very_low'])
rule1.view()

# Create control system
capability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5])
capability = ctrl.ControlSystemSimulation(capability_ctrl)


# Compute output
capability.compute()
# Print output
print("Overall Capability:", capability.output['Overall Capability'])

overall_capability.view(sim=capability)
