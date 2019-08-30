import numpy as np
import tester
import os

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


'''you can run this to see trivial output but make sure that you uncommented following line and delete the next one (you may also want to change 'create_video' to False) '''

# times = np.loadtxt('../data/test_times.txt', dtype='int')
times = [1554105954]
model = '1_cluster_9_periods'
# model = 'WHyTeS'
results = []
edges_of_cell = [0.5, 0.5]
speed = 1.

try:
    os.mkdir('../results/' + str(model))
except OSError as error:
    pass

file_path = '../results/' + str(model) + '/output.txt'
if os.path.exists(file_path):
    os.remove(file_path)

for time in times:
    path_model = '../models/' + model + '/' + str(time) + '_model.txt'
    test_data_path = '../data/time_windows/' + str(time) + '_test_data.txt'
    print time

    result = tester.test_model(path_model=path_model, path_data=test_data_path, testing_time=time, model_name=model, edges_of_cell=edges_of_cell, speed=speed, create_video=False)
    results.append(result)
    with open(file_path, 'a') as file:
        file.write(str(result))
