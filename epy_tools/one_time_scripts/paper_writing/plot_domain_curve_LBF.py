import os

import numpy as np

# Read all the npz file in
file_dir = '/media/ppo/workspace/epymarl_research/output_each_run_data'

# The filename contain the comb information
# The filename is in the format of: `ExpVarComb(key1=value1,key2=value2,...).npz`
# The key-value pairs are separated by ", " and the whole string is enclosed by "()"
filename_to_npz = {}
for filename in os.listdir(file_dir):
    if filename.endswith('.npz'):
        comb = filename.split('.npz')[0]
        filename_to_npz[comb] = np.load(os.path.join(file_dir, filename))

filename_to_runs_y = {}
filename_to_runs_x = {}
SIGHT_X = filename_to_npz[list(filename_to_npz.keys())[0]]['sight_x'][0]
N_DATA_PER_RUN = len(SIGHT_X)
for filename, npz in filename_to_npz.items():
    filename_to_runs_y[filename] = npz['sight_y']
    filename_to_runs_x[filename] = npz['sight_x']

# LBF: sight=6s,10s 一群；其它另一群。共2群
N_RUNS = 5
N_TASKS = 6
group1_y = [[] for _ in range(N_RUNS)]
group1_x = [[] for _ in range(N_RUNS)]
group2_y = [[] for _ in range(N_RUNS)]
group2_x = [[] for _ in range(N_RUNS)]

for filename, runs_y in filename_to_runs_y.items():
    if 'sight=6s' in filename or 'sight=10s' in filename:
        for run_idx in range(N_RUNS):
            group1_y[run_idx].append(runs_y[run_idx])
            group1_x[run_idx].append(filename_to_runs_x[filename][run_idx])
    else:
        for run_idx in range(N_RUNS):
            group2_y[run_idx].append(runs_y[run_idx])
            group2_x[run_idx].append(filename_to_runs_x[filename][run_idx])

without_dsr_y = np.array(group1_y)
without_dsr_x = np.array(group1_x)
with_dsr_y = np.array(group2_y)
with_dsr_x = np.array(group2_x)

# 現在 shape 為 [N_RUNS, N_TASKS, N_DATA_PER_RUN]
# 併成 [N_RUNS * N_TASKS, N_DATA_PER_RUN]
without_dsr_y = without_dsr_y.reshape(-1, N_DATA_PER_RUN)
without_dsr_x = without_dsr_x.reshape(-1, N_DATA_PER_RUN)
with_dsr_y = with_dsr_y.reshape(-1, N_DATA_PER_RUN)
with_dsr_x = with_dsr_x.reshape(-1, N_DATA_PER_RUN)

# Calculate the mean and std
without_dsr_y_mean = np.mean(without_dsr_y, axis=0)
without_dsr_y_std = np.std(without_dsr_y, axis=0)
with_dsr_y_mean = np.mean(with_dsr_y, axis=0)
with_dsr_y_std = np.std(with_dsr_y, axis=0)
# x 也要算 mean
without_dsr_x_mean = np.mean(without_dsr_x, axis=0)
with_dsr_x_mean = np.mean(with_dsr_x, axis=0)

# Compute 95% confidence interval
import scipy.stats as stats

confidence = 0.95
n = without_dsr_y.shape[0]
h = without_dsr_y_std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
without_dsr_y_ci = h
without_dsr_y_ci = np.array(without_dsr_y_ci)
n = with_dsr_y.shape[0]
h = with_dsr_y_std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
with_dsr_y_ci = h
with_dsr_y_ci = np.array(with_dsr_y_ci)

# Compute IQM (Interquartile Mean)
# Remove outliers (25% and 75% quantile) and compute mean
n_to_remove = int(without_dsr_y.shape[0] * 0.25)
without_dsr_y_iqm = np.mean(np.sort(without_dsr_y, axis=0)[n_to_remove:-n_to_remove], axis=0)
n_to_remove = int(with_dsr_y.shape[0] * 0.25)
with_dsr_y_iqm = np.mean(np.sort(with_dsr_y, axis=0)[n_to_remove:-n_to_remove], axis=0)




# Plot mean+-std
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.figure(figsize=(6,6))
plt.plot(without_dsr_x_mean, without_dsr_y_mean, label='QMIX w/o DSR', color='black')
plt.fill_between(without_dsr_x_mean, without_dsr_y_mean - without_dsr_y_std, without_dsr_y_mean + without_dsr_y_std, alpha=0.1, color='black')
plt.plot(with_dsr_x_mean, with_dsr_y_mean, label='QMIX w/   DSR (ours)', color='red')
plt.fill_between(with_dsr_x_mean, with_dsr_y_mean - with_dsr_y_std, with_dsr_y_mean + with_dsr_y_std, alpha=0.1,color='red')
plt.legend()
plt.title('QMIX w/ and w/o DSR (mean+-std)')
plt.show()

# Plot iqm+-ci
plt.figure(figsize=(6,6))
plt.plot(without_dsr_x_mean, without_dsr_y_iqm, label='QMIX w/o DSR', color='black')
plt.fill_between(without_dsr_x_mean, without_dsr_y_iqm - without_dsr_y_ci, without_dsr_y_iqm + without_dsr_y_ci, alpha=0.1, color='black')
plt.plot(with_dsr_x_mean, with_dsr_y_iqm, label='QMIX w/   DSR (ours)', color='red')
plt.fill_between(with_dsr_x_mean, with_dsr_y_iqm - with_dsr_y_ci, with_dsr_y_iqm + with_dsr_y_ci, alpha=0.1, color='red')
plt.legend()
plt.title('QMIX w/ and w/o DSR (iqm+-ci)')
plt.show()


# Plot mean+-ci
plt.figure(figsize=(6,6))
plt.plot(without_dsr_x_mean, without_dsr_y_mean, label='QMIX w/o DSR', color='black')
plt.fill_between(without_dsr_x_mean, without_dsr_y_mean - without_dsr_y_ci, without_dsr_y_mean + without_dsr_y_ci, alpha=0.1, color='black')
plt.plot(with_dsr_x_mean, with_dsr_y_mean, label='QMIX w/   DSR (ours)', color='red')
plt.fill_between(with_dsr_x_mean, with_dsr_y_mean - with_dsr_y_ci, with_dsr_y_mean + with_dsr_y_ci, alpha=0.1, color='red')
plt.legend()
plt.title('QMIX w/ and w/o DSR (iqm+-ci)')
plt.show()

