import numpy as np
import matplotlib.pyplot as plt
import os

def summarize(text_file_to_analyze):
    data_load = np.loadtxt(text_file_to_analyze)
    # ordering the data
    all_kills = data_load[:, -1] * (-1)  # number of kills using chosen model in each scenario
    data = np.empty_like(data_load)
    np.copyto(data, data_load[all_kills.argsort(kind='mergesort'), :])
    # number of kills
    clockwise_default_kills = np.sum(data[:, 2])
    clockwise_model_kills = np.sum(data[:, 4])
    counter_clockwise_default_kills = np.sum(data[:, 3])
    counter_clockwise_model_kills = np.sum(data[:, 5])
    model_chosen_kills = np.sum(data[:, -1])
    # ordering kills by the weights
    ordering_by_lower_mean_weight = np.min(data[:, [-3, -2]],
                                           axis=1)  # choosing the lower of the two mean weghts of two possible paths for every scenario
    worst_scenario_ordered_kills = data[
        ordering_by_lower_mean_weight.argsort(kind='mergesort'), -1]  # ordering kills by weights
    # plot
    plt.plot(worst_scenario_ordered_kills)
    plt.title('kills ordered by the mean weight of selected paths')
    plt.savefig(text_file_to_analyze + '.png')
    plt.close()
    # calculation of the ratio
    part_kills = 0
    total_kills = np.sum(worst_scenario_ordered_kills)
    for idx, kills in enumerate(worst_scenario_ordered_kills):
        if part_kills + kills > 20:
            print(text_file_to_analyze)
            print('ratio of refused walks needed to kill less then 20 people during the whole day: ' + str(
                1.0 - (float(idx) / float(len(worst_scenario_ordered_kills)))))
            print('clockwise_default_kills: ' + str(clockwise_default_kills))
            print('clockwise_model_kills: ' + str(clockwise_model_kills))
            print('counter_clockwise_default_kills: ' + str(counter_clockwise_default_kills))
            print('counter_clockwise_model_kills: ' + str(counter_clockwise_model_kills))
            print('model_chosen_kills: ' + str(model_chosen_kills))
            print('')
            break
        else:
            part_kills += kills

if __name__ == "__main__":
    with open("list_of_outputs.txt", "r") as ins:
        list_of_outputs = []
        for line in ins:
            list_of_outputs.append(line.rstrip())

    for text_file_to_analyze in list_of_outputs:
    #for text_file_to_analyze in ['2_clusters_0_periodicities_output_test.txt']:
        data_load = np.loadtxt(text_file_to_analyze)
        # ordering the data
        all_kills = data_load[:, -1] * (-1)  # number of kills using chosen model in each scenario
        data = np.empty_like(data_load)
        np.copyto(data, data_load[all_kills.argsort(kind='mergesort'), :])
        # number of kills
        clockwise_default_kills = np.sum(data[:,2])
        clockwise_model_kills = np.sum(data[:,4])
        counter_clockwise_default_kills = np.sum(data[:,3])
        counter_clockwise_model_kills = np.sum(data[:,5])
        model_chosen_kills = np.sum(data[:,-1])
        # ordering kills by the weights
        ordering_by_lower_mean_weight = np.min(data[:, [-3, -2]], axis=1)  # choosing the lower of the two mean weghts of two possible paths for every scenario
        worst_scenario_ordered_kills = data[ordering_by_lower_mean_weight.argsort(kind='mergesort'), -1]  # ordering kills by weights
        # plot
        plt.plot(worst_scenario_ordered_kills)
        plt.title('kills ordered by the mean weight of selected paths')
        plt.savefig(text_file_to_analyze + '.png')
        plt.close()
        # calculation of the ratio
        part_kills = 0
        total_kills = np.sum(worst_scenario_ordered_kills)
        for idx, kills in enumerate(worst_scenario_ordered_kills):
            if part_kills + kills > 20:
                print(text_file_to_analyze)
                print('ratio of refused walks needed to kill less then 20 people during the whole day: ' + str(1.0 - (float(idx)/float(len(worst_scenario_ordered_kills)))))
                print('clockwise_default_kills: ' + str(clockwise_default_kills))
                print('clockwise_model_kills: ' + str(clockwise_model_kills))
                print('counter_clockwise_default_kills: ' + str(counter_clockwise_default_kills))
                print('counter_clockwise_model_kills: ' + str(counter_clockwise_model_kills))
                print('model_chosen_kills: ' + str(model_chosen_kills))
                print('')
                break
            else:
                part_kills += kills

