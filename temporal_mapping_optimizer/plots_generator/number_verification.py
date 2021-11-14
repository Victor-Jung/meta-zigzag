import os
import sys
import ast
import numpy as np

### Resnet Even ###

resnet_even_timeloop_time_list = [86,468,224,343,254,304,181,195]
resnet_even_salsa_time_list = [8.705, 10.622, 8.943, 9.96, 8.974, 10.98, 8.058, 9.134]

resnet_even_timeloop_energy_list = [513360691.20000005, 430052474.88000005, 218494402.56, 424272199.68, 232367063.04, 457797795.84, 338724126.72, 673980088.32]
resnet_even_salsa_energy_list = [489757900.8, 427740364.8000001, 218494402.56, 418491924.48, 232367063.04, 457797795.84, 335255961.6, 668199813.12]

### Alexnet and Resnet Uneven ###

# Execution time data

resnet_uneven_timeloop_time_list = [86,468,224,343,254,304,181,195]
alexnet_uneven_timeloop_time_list = [89,230,506,522,395]

resnet_uneven_salsa_time_list = [9.549, 10.622, 9.757, 10.668, 10.205, 10.647, 10.346, 10.679]
alexnet_uneven_salsa_time_list = [9.287, 10.854, 10.549, 9.999, 9.11]

# Energy data

resnet_uneven_timeloop_energy_list = [513360691.20000005, 430052474.88000005, 218494402.56, 424272199.68, 232367063.04, 457797795.84, 338724126.72, 673980088.32]
resnet_uneven_salsa_energy_list = [422360015.5, 394193368.1, 201959039.3, 391400471.4, 217458115.60000002, 428657427.5, 328777373.6, 654425345.0]

alexnet_uneven_timeloop_energy_list = [378120314.88, 2668523520.0, 680317747.2, 466503598.08000004, 309507194.88]
alexnet_uneven_salsa_energy_list = [316242181.4, 673423628.1999999, 607709258.5, 447532898.1, 300469257.2]

# Arithmetic mean for ResNet34 Even mapping energy

resnet_even_timeloop_energy_mean = sum(resnet_even_timeloop_energy_list)/8
resnet_even_salsa_energy_mean = sum(resnet_even_salsa_energy_list)/8

print("resnet_even_timeloop_energy_mean: ", resnet_even_timeloop_energy_mean)
print("resnet_even_salsa_energy_mean: ", resnet_even_salsa_energy_mean)

print("resnet_even_timeloop_energy_mean/resnet_even_salsa_energy_mean: ", resnet_even_timeloop_energy_mean/resnet_even_salsa_energy_mean)
print("Improvement of ", ((resnet_even_timeloop_energy_mean/resnet_even_salsa_energy_mean)-1)*100," %\n")

# Arithmetic mean for Alexnet and ResNet34 Uneven mapping energy

resnet_alexnet_uneven_timeloop_energy_mean = sum(resnet_uneven_timeloop_energy_list + alexnet_uneven_timeloop_energy_list)/(8+5)
resnet_alexnet_uneven_salsa_energy_mean = sum(resnet_uneven_salsa_energy_list + alexnet_uneven_salsa_energy_list)/(8+5)

print("resnet_alexnet_uneven_timeloop_energy_mean: ", resnet_alexnet_uneven_timeloop_energy_mean)
print("resnet_alexnet_uneven_salsa_energy_mean: ", resnet_alexnet_uneven_salsa_energy_mean)

print("resnet_alexnet_uneven_timeloop_energy_mean/resnet_alexnet_uneven_salsa_energy_mean: ", resnet_alexnet_uneven_timeloop_energy_mean/resnet_alexnet_uneven_salsa_energy_mean)
print("Improvement of ", ((resnet_alexnet_uneven_timeloop_energy_mean/resnet_alexnet_uneven_salsa_energy_mean)-1)*100," %\n")

# Now for the energy we consider both uneven and even benchmark since the even constraint doesn't affect the search time

resnet_alexnet_timeloop_time_mean = sum(resnet_even_timeloop_time_list + resnet_uneven_timeloop_time_list + alexnet_uneven_timeloop_time_list)/(8+8+5)
resnet_alexnet_salsa_time_mean = sum(resnet_even_salsa_time_list + resnet_uneven_salsa_time_list + alexnet_uneven_salsa_time_list)/(8+8+5)

print("resnet_alexnet_timeloop_time_mean: ", resnet_alexnet_uneven_timeloop_energy_mean)
print("resnet_alexnet_salsa_time_mean: ", resnet_alexnet_uneven_salsa_energy_mean)

print("resnet_alexnet_timeloop_time_mean/resnet_alexnet_salsa_time_mean: ", resnet_alexnet_timeloop_time_mean/resnet_alexnet_salsa_time_mean)
print("Improvement of ", ((resnet_alexnet_timeloop_time_mean/resnet_alexnet_salsa_time_mean)-1)*100," %\n")

# For uneven mapping ResNet34 and AlexNet: total energy * total_time (Energy Time Product: ETP)
# Duplication in Resnet34 L1: 1 L2: 7 L9: 2 L10: 7 L18: 2 L19: 11 L31: 2 L32: 5
# 38 Layer in total in ResNet34 including 1 Fully connected

resnet_uneven_timeloop_energy_list = [513360691.20000005 * 1, 430052474.88000005 * 7, 218494402.56 * 2, 424272199.68 * 7, 232367063.04 * 2, 457797795.84 * 11, 338724126.72 * 2, 673980088.32 * 5]
resnet_uneven_salsa_energy_list = [422360015.5 * 1, 394193368.1 * 7, 201959039.3 * 2, 391400471.4 * 7, 217458115.60000002 * 2, 428657427.5 * 11, 328777373.6 * 2, 654425345.0 * 5]

resnet_uneven_timeloop_time_list = [86 * 1, 468 * 7, 224 * 2, 343 * 7, 254 * 2, 304 * 11, 181 * 2, 195 * 5]
resnet_uneven_salsa_time_list = [9.549 * 1, 10.622 * 7, 9.757 * 2, 10.668 * 7, 10.205 * 2, 10.647 * 11, 10.34 * 26, 10.679 * 5]

resnet_timeloop_ETP = sum(resnet_uneven_timeloop_energy_list) * sum(resnet_uneven_timeloop_time_list)
resnet_salsa_ETP = sum(resnet_uneven_salsa_energy_list) * sum(resnet_uneven_salsa_time_list)

print("resnet_timeloop_ETP/resnet_salsa_ETP :", resnet_timeloop_ETP/resnet_salsa_ETP)

alexnet_uneven_timeloop_energy_list = [378120314.88, 822362112.0, 680317747.2, 466503598.08000004, 309507194.88]
alexnet_uneven_salsa_energy_list = [316242181.4, 673423628.1999999, 607709258.5, 447532898.1, 300469257.2]

alexnet_uneven_timeloop_time_list = [89, 230, 453, 522, 395]
alexnet_uneven_salsa_time_list = [9.287, 10.854, 10.549, 9.999, 9.11]

alexnet_timeloop_ETP = sum(alexnet_uneven_timeloop_energy_list) * sum(alexnet_uneven_timeloop_time_list)
alexnet_salsa_ETP = sum(alexnet_uneven_salsa_energy_list) * sum(alexnet_uneven_salsa_time_list)

print("alexnet_timeloop_ETP/alexet_salsa_ETP :", alexnet_timeloop_ETP/alexnet_salsa_ETP, "\n")

resnet_uneven_timeloop_total_energy = sum(resnet_uneven_timeloop_energy_list)
resnet_uneven_salsa_total_energy = sum(resnet_uneven_salsa_energy_list)

print("resnet_uneven_timeloop_total_energy/resnet_uneven_salsa_total_energy :", resnet_uneven_timeloop_total_energy/resnet_uneven_salsa_total_energy)
print("Improvement of ", (1-(resnet_uneven_salsa_total_energy/resnet_uneven_timeloop_total_energy))*100," %\n")

alexnet_uneven_timeloop_total_energy = sum(alexnet_uneven_timeloop_energy_list)
alexnet_uneven_salsa_total_energy = sum(alexnet_uneven_salsa_energy_list)

print("alexnet_uneven_timeloop_total_energy/alexnet_uneven_salsa_total_energy :", alexnet_uneven_timeloop_total_energy/alexnet_uneven_salsa_total_energy)
print("Improvement of ", (1-(alexnet_uneven_salsa_total_energy/alexnet_uneven_timeloop_total_energy))*100," %\n")

resnet_uneven_timeloop_total_time = sum(resnet_uneven_timeloop_time_list)
resnet_uneven_salsa_total_time = sum(resnet_uneven_salsa_time_list)

print("resnet_uneven_timeloop_total_time/resnet_uneven_salsa_total_time :", resnet_uneven_timeloop_total_time/resnet_uneven_salsa_total_time)
print("Improvement of ", ((resnet_uneven_timeloop_total_time/resnet_uneven_salsa_total_time)-1)*100," %\n")

alexnet_uneven_timeloop_total_time = sum(alexnet_uneven_timeloop_time_list)
alexnet_uneven_salsa_total_time = sum(alexnet_uneven_salsa_time_list)

print("alexnet_uneven_timeloop_total_time/alexnet_uneven_salsa_total_time :", alexnet_uneven_timeloop_total_time/alexnet_uneven_salsa_total_time)
print("Improvement of ", ((alexnet_uneven_timeloop_total_time/alexnet_uneven_salsa_total_time)-1)*100," %\n")

print("Alexnet uneven :", 1 - (alexnet_uneven_salsa_total_energy/alexnet_uneven_timeloop_total_energy))
print("Resnet uneven :", 1 - (resnet_uneven_salsa_total_energy/resnet_uneven_timeloop_total_energy))

