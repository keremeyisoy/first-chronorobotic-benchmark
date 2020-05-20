import numpy as np
import tester
#import summarize
import summarize_new
import os

import path_finder_new as pf

"""
Before run this, please create new directories with your models name inside following directories; 'models' and 'results'
and, change the variable 'model' with the same name of the directories you created.

You can find the list of positions in '../data/positions.txt' (x, y, angle)
Model outputs should be same as this file with an additional column of weights (order of rows is not important for testing method).

Here, are the parameters of grid;

edges of cells: x...0.5 [m], y...0.5 [m], angle...pi/4.0 [rad]
number of cells: x...24, y...33, angles...8
center of "first" cell: (-9.5, 0.25, -3.0*pi/4.0)
center of "last" cell: (2.0, 16.25, pi) 

If you change the argument 'create_video' to True, there will be video of every time window in results

outputs will be written in ../results/$model/output.txt in following format;
list of values; [testing_time, number_of_detections_in_testing_data, interactions_of_dummy_model_clockwise, interactions_of_dummy_model_counterclockwise, interactions_of_real_model_clockwise, interactions_of_real_model_counterclockwise, total_weight_in_clockwise, total_weight_in_counterclockwise, total_interactions_of_chosen_trajectory]

Since this code is prepared in a short time for scientific reasons, sorry in advance for any ambiguity
"""



tester = tester.Tester(radius_of_robot=1.)

'''you can run this to see trivial output. If you want to run this for your model, make sure that you uncommented following line and delete the next one (you may also want to change 'create_video' to False) '''

times = np.loadtxt('../data/test_times.txt', dtype='int')
#times = np.loadtxt('../data/test_times_winter.txt', dtype='int')
#times = [1554092829, 1554105954]

# individual models
#models = ['published_ral_3_clusters_3_periodicities']  # calculated by buggy code
#models = ['published_cliffmap_model_pq']
#models = ['published_predictions_stef_euc_o2']
#models = ['ral_variant_3_clusters_2_periodicities']  # possible to calculate, repaired bug in fremen
#models = ['ral_variant_3_clusters_2_periodicities', 'test_3_clusters_2_periodicities', 'test2_3_clusters_2_periodicities', 'test3_3_clusters_2_periodicities']

# models, we need to compare
#models = ['published_ral_3_clusters_3_periodicities', 'published_cliffmap_model_pq', 'published_predictions_stef_euc_o2', 'ral_variant_3_clusters_2_periodicities']
#models = ['published_ral_3_clusters_3_periodicities', 'published_cliffmap_model_pq', 'published_predictions_stef_euc_o2', 'ral_variant_3_clusters_2_periodicities', 'test3_3_clusters_2_periodicities']
#models = ['test_speedstep01_3_clusters_2_periodicities']
#models = ['cost_test_minus_speed_3_clusters_2_periodicities']
#models = ['cost_exp_3_clusters_2_periodicities']
#models = ['cost_pow2_3_clusters_2_periodicities']
#models = ['cost_exp_F_3_clusters_2_periodicities']
#models = ['cost_F_3_clusters_3_periodicities']
#models = ['cost_winter_3_clusters_3_periodicities']
#models = ['cost_chiF_3_clusters_5_periodicities']
#models = ['cost_chiF_10_clusters_4_periodicities']
#models = ['integral_s01_10_clusters_4_periodicities']
#models = ['chiF_1_clusters_4_periodicities']
#models = ['chiF_cond_Ceq15_5_clusters_4_periodicities']
#models = ['chiF_1_clusters_0_periodicities', 'chiF_2_clusters_0_periodicities', 'chiF_3_clusters_0_periodicities', 'chiF_1_clusters_1_periodicities', 'chiF_2_clusters_1_periodicities', 'chiF_3_clusters_1_periodicities', 'chiF_1_clusters_2_periodicities', 'chiF_2_clusters_2_periodicities', 'chiF_3_clusters_2_periodicities', 'chiF_1_clusters_3_periodicities', 'chiF_2_clusters_3_periodicities', 'chiF_3_clusters_3_periodicities', 'chiF_1_clusters_4_periodicities', 'chiF_2_clusters_4_periodicities', 'chiF_3_clusters_4_periodicities']
models = ['published_ral_3_clusters_3_periodicities', 'published_cliffmap_model_pq', 'published_predictions_stef_euc_o2', 'ral_variant_3_clusters_2_periodicities', 'model_daily_histogram', 'model_segment_means', 'model_weekly_histogram', 'model_prophet']
models = ['RAL2020_3_clusters_3_periodicities']
#models = ['occ_grid']


result_dir = 'testing_speed'
model_dir = '/home/tom/pro/my/whyte_branches/2020_test_for_icra_RAL/models'


edges_of_cell = [0.5, 0.5]
speed = 1.

for model in models:
    #break # delete me :)
    print 'testing  ' + model
    # creating path for the outputs of planner
    try:
        os.mkdir('../results/' + result_dir)
    except OSError as error:
        pass

    output_path = '../results/' + result_dir + '/' + str(model) + '_output.txt'
    if os.path.exists(output_path):
        os.remove(output_path)

    for time in times:
        path_model = model_dir + '/' + str(model) + '/' + str(time) + '_model.txt'
        #path_model = model_dir + '/' + str(model) + '/' + str(time) + '_model.txt.npy'
        test_data_path = '../data/time_windows/' + str(time) + '_test_data.txt'
        #test_data_path = '../data/time_windows_winter/' + str(time) + '_test_data.txt'
        result = tester.test_model(path_model=path_model, path_data=test_data_path, testing_time=time, model_name=model, edges_of_cell=edges_of_cell, speed=speed, create_video=False)
        with open(output_path, 'a') as file:
            file.write(' '.join(str(value) for value in result) + '\n')

for model in models:

    print '\n statistics of ' + model
    output_path = '../results/' + result_dir + '/' + str(model) + '_output.txt'
    #summarize.summarize(output_path)
    summarize_new.summarize(output_path)
