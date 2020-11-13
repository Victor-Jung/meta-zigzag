# The "half" of AlexNet, reported in Eyeriss paper, which we used to benchmark against.
layer_info = \
{1: {'B': 1, 'K': 96, 'C': 3, 'OY': 56, 'OX': 56, 'FY': 11, 'FX': 11, 'SY': 4, 'SX': 4, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 2: {'B': 1, 'K': 256, 'C': 48, 'OY': 26, 'OX': 26, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 3: {'B': 1, 'K': 384, 'C': 256, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 4: {'B': 1, 'K': 384, 'C': 192, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 5: {'B': 1, 'K': 256, 'C': 192, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 6: {'B': 1, 'K': 4096, 'C': 256, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 7: {'B': 1, 'K': 4096, 'C': 4096, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
 8: {'B': 1, 'K': 1000, 'C': 4096, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},

 # an example of defining group convolution, in which 'K' and 'C' are total amount for all groups.
 9: {'B': 1, 'K': 256, 'C': 192, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 2}}





# The complete of AlexNet, reported in AlexNet paper.
# layer_info = \
# {1: {'B': 1, 'K': 96, 'C': 3, 'OY': 56, 'OX': 56, 'FY': 11, 'FX': 11, 'SY': 4, 'SX': 4, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  2: {'B': 1, 'K': 256, 'C': 96, 'OY': 26, 'OX': 26, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  3: {'B': 1, 'K': 384, 'C': 256, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  4: {'B': 1, 'K': 384, 'C': 384, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  5: {'B': 1, 'K': 256, 'C': 384, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  6: {'B': 1, 'K': 4096, 'C': 256, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  7: {'B': 1, 'K': 4096, 'C': 4096, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1},
#  8: {'B': 1, 'K': 1000, 'C': 4096, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1}}