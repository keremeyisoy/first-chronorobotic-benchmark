import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(6666)

def summarize(text_file_to_analyze):
    # load of outputs
    data_load = np.loadtxt(text_file_to_analyze)
    data = np.empty_like(data_load)
    # ordering the rows randomly (to simulate the random choice when the weights are identical for different times)
    np.copyto(data, data_load[np.random.rand(len(data)).argsort(kind='mergesort'), :])
    # ordering kills by the weights
    ordering_by_lower_mean_weight = np.min(data[:, [-3, -2]], axis=1)  
    ordered_kills = data[ordering_by_lower_mean_weight.argsort(kind='mergesort'), -1]  # ordering kills by weights
    cummulative_kills = np.cumsum(ordered_kills)
    print('expected service social cost: ' + str(np.mean(cummulative_kills)))
    print('service social cost, servicing ratio 100%: ' + str(cummulative_kills[-1]))
    print('service social cost, servicing ratio 90%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.9)]))
    print('service social cost, servicing ratio 50%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.5)]))
    print('expected encounters: ' + str(np.sum(cummulative_kills)/len(cummulative_kills)))

    plt.plot(cummulative_kills)
    plt.savefig(text_file_to_analyze + '.png')
    plt.close()
    # return


"""

    # POKUS SPECIFICKY PRO DATA !!!
    part_len = 20  # this times 40 seconds
    part_data = np.empty_like(data_load)[:part_len, :]
    new_way_cum = np.zeros(part_len)
    #for i in xrange(6):
    cont = True
    i = -1
    while cont == True:
        i += 1
        if (i+1)*part_len > len(data):
            print(i)
            #print((i+1)*part_len)
            #print(len(data))
            part_data = np.zeros_like(data_load)[:part_len, :]
            part_data[(i+1)*part_len - len(data):, :] += data[i*part_len:, :]
            cont = False
        else:
            np.copyto(part_data, data[i*part_len:(i+1)*part_len, :])
        # ordering kills by the weights
        ordering = np.min(part_data[:, [-3, -2]], axis=1)  
        ordered = data[ordering.argsort(kind='mergesort'), -1]  # ordering kills by weights
        new_way_cum += ordered
    cummulative_kills = np.cumsum(new_way_cum)
    print('improved expected service social cost: ' + str(np.mean(cummulative_kills)))
    print('improved service social cost, servicing ratio 100%: ' + str(cummulative_kills[-1]))
    print('improved service social cost, servicing ratio 90%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.9)]))
    print('improved service social cost, servicing ratio 50%: ' + str(cummulative_kills[int(len(cummulative_kills)*0.5)]))

"""
