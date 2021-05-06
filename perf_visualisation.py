import os
import yaml
import subprocess
import matplotlib.pyplot as plt

loma_lpf_limit = 9
settings_file_path = "inputs/settings.yaml"
data_file_path = "visualisation_data.yaml"
run_zigzag_command = "python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml"

with open(settings_file_path) as f:
    settings_doc = yaml.safe_load(f)

# Execute MCMC and store result in settings doc
settings_doc["temporal_mapping_search_method"] = "RL"
with open(settings_file_path, "w") as f:
    yaml.dump(settings_doc, f)

process = subprocess.Popen(run_zigzag_command.split(), stdout=subprocess.PIPE, text=True)
output, error = process.communicate()
print(output)

# Load MCMC utilization list and lpf range and create loma lpf range
with open(data_file_path) as f:
     data_doc = yaml.safe_load(f)

mcmc_utilization_list = data_doc["mcmc_utilization_list"]
mcmc_exec_time_list = data_doc["mcmc_exec_time_list"]
lpf_range = data_doc["lpf_range"]

# Reset loma utilization array
data_doc["loma_utilization_list"] = []
data_doc["loma_exec_time_list"] = []
with open(data_file_path, "w") as f:
        yaml.dump(data_doc, f)

# Execute loma for the lpf_range of MCMC until 10 lpf
if lpf_range[0] <= loma_lpf_limit:

    if lpf_range[-1] >= loma_lpf_limit:
        loma_lpf_range = [*range(lpf_range[0], loma_lpf_limit + 1)]
    else:
        loma_lpf_range = lpf_range

    settings_doc["temporal_mapping_search_method"] = "loma"

    with open(settings_file_path, "w") as f:
        yaml.dump(settings_doc, f)

    for lpf in loma_lpf_range:

        settings_doc["max_nb_lpf_layer"] = lpf
        with open(settings_file_path, "w") as f:
            yaml.dump(settings_doc, f)

        process = subprocess.Popen(run_zigzag_command.split(), stdout=subprocess.PIPE, text=True)
        output, error = process.communicate()
        print(output, error)

# Get loma results
with open(data_file_path) as f:
     data_doc = yaml.safe_load(f)
loma_utilization_list = data_doc["loma_utilization_list"]
loma_exec_time_list = data_doc["loma_exec_time_list"]

fig, ax1 = plt.subplots()
#fig.title('Loma And MCMC performances')

ax1.set_xlabel("Temporal Mapping Size")
ax1.set_ylabel("Best Utilization Found")
ax1.plot(loma_lpf_range, loma_utilization_list, label='loma')
ax1.plot(lpf_range, mcmc_utilization_list, label='mcmc')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Execution Time (s)')
ax2.bar(loma_lpf_range, loma_exec_time_list, label='loma', width = 0.25, alpha=0.75)
ax2.bar(lpf_range, mcmc_exec_time_list, label='mcmc', width = 0.25, alpha=0.75)
ax2.tick_params(axis='y')

fig.tight_layout()
plt.legend()
plt.show()