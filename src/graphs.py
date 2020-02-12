import numpy as np
from scipy.stats import ttest_rel

list_of_outputs = ['dummy_output.txt', 'cliffmap_model_pq_output.txt', 'repaired_3_clusters_3_periodicities_output.txt', 'stef_euc_o2_output.txt']
labels = ['dummy', 'CLiFF', 'WHyTe', 'STeF']

import matplotlib.pyplot as plt

np.random.seed(6666)

values = []
for idx, text_file_to_ananlyze in enumerate(list_of_outputs):
    # load of outputs
    data_load = np.loadtxt(text_file_to_ananlyze)
    data = np.empty_like(data_load)
    # ordering the rows randomly (to simulate the random choice when the weights are identical for different times)
    np.copyto(data, data_load[np.random.rand(len(data)).argsort(kind='mergesort'), :])
    # ordering kills by the weights
    ordering_by_lower_mean_weight = np.min(data[:, [-3, -2]], axis=1)  
    ordered_kills = data[ordering_by_lower_mean_weight.argsort(kind='mergesort'), -1]  # ordering kills by weights
    cummulative_kills = np.cumsum(ordered_kills)
    print(labels[idx])
    print('90%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.9)]))
    print('50%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.5)]))
    # values... for later use (ttest, sum of cumsum)
    values.append(cummulative_kills)
    # plot
    plt.plot(cummulative_kills, label=labels[idx])
# redefine x axis from number of scenarios to percentage
x = np.arange(11)*120
xi = np.arange(11)*10
plt.xticks(x, xi)
# some titles, need to be changed
plt.title('cummulative number of encounters\n ordered by the mean weight of selected paths')
plt.legend()
plt.xlabel('service tasks percentage')
plt.ylabel('cummulative number of encounters')
plt.savefig('cummulative_kills.eps')
plt.close()

# ttest between STeFF and WHyTe
print(ttest_rel(values[1], values[0])[1])
print(ttest_rel(values[3], values[2])[1])
# sum of cumsum (japanese porn with very fat actors)
print('dummy: ' + str(np.sum(values[0])))
print('CLiFF: ' + str(np.sum(values[1])))
print('WHyTe: ' + str(np.sum(values[2])))
print('STeF: ' + str(np.sum(values[3])))


