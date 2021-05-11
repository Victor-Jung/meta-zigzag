import numpy as np

loop_types_list = ["FX", "FY", "OX", "OY", "C", "K", "B"]
# Corresponding number for each loop_type {"FX": 1, "FY": 2, "OX": 3, "OY": 4, "C": 5, "K": 6, "B": 7}
loop_type_to_ids = {key: value + 1 for value, key in enumerate(loop_types_list)}
# Corresponding number for each loop_type: {1: "FX", 2: "FY", 3: "OX", 4: "OY", 5: "C", 6: "K", 7: "B"}
ids_to_loop_type = {value: key for key, value in loop_type_to_ids.items()}

operand_cost_types = ["W", "I", "O"]
operand_cost_template = {key: [] for key in operand_cost_types}
