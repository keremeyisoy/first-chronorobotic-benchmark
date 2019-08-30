"""
this is an example, how to run the method
"""
import directions
import fremen
import tester as tm
import numpy as np
from time import time

# parameters for the method
number_of_clusters = 20
#number_of_spatial_dimensions = 2  # known from data
number_of_spatial_dimensions = 4  # france data
#list_of_periodicities = [21600.0, 43200.0, 86400.0, 86400.0*7.0]  # the most prominent periods, found by FreMEn
list_of_periodicities = [4114.285714285715, 10800.0, 5400.0, 4800.0, 604800.0, 86400.0]  # the most prominent periods, found by FreMEn
#list_of_periodicities = []
#list_of_periodicities = [86400.0]  # the most prominent periods, found by FreMEn
#movement_included = True  # True, if last two columns of dataset are phi and v, i.e., the angle and speed of human.
movement_included = False  # using velocity vector precalculated in dataset.

#structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
"""
TS = np.loadtxt('../data/training_dataset.txt')[:, [0,-1]]
start = time()
P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=fremen.build_frequencies(60*60*24, 60*60), weights=1.0, return_all=True)
finish = time()
print('time to run fremen: ' + str(finish-start))
print(P)
"""

structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
# load and train the predictor
start = time()
dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
dirs = dirs.fit('../data/training_dataset.txt')
#dirs = dirs.fit('../data/training_03_04_rotated_speeds.txt')
finish = time()
print('time to create model: ' + str(finish-start))
# self.C, self.Pi, self.PREC
#print('C: ' + str(dirs.C))
#print('PREC: ' + str(dirs.PREC))

"""
start = time()
print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/training_03_04_rotated_speeds.txt')))
finish = time()
print('time to calculate RMSE: ' + str(finish-start))

# predict values from dataset
# first transform data and get target values
#X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
X, target = dirs.transform_data('../data/test_dataset.txt')
# than predict values
prediction = dirs.predict(X)
# now, you can compare target and prediction in any way, for example RMSE
print('manually calculated RMSE: ' + str(np.sqrt(np.mean((prediction - target) ** 2.0))))

# or calculate RMSE of prediction of values directly
#print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')))
print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/test_dataset.txt')))
"""

start = time()
#out = dirs.model_to_directions_for_kevin_no_time_dimension()
#model_time = 1554105948
model_time = 1554139930
out = dirs.model_to_directions(model_time)
#np.savetxt('../data/model_of_8angles_0.5m_over_month.txt', out)
np.savetxt('../data/' + str(model_time) + '_model.txt', out)
finish = time()
print('time to save model for specific time: ' + str(finish-start))
#print(np.sum(out[:,-1]))


#tester = tm.Tester()
tester = tm.Tester(radius_of_robot=1.)

edges_of_cell = [3600., 0.5, 0.5]
print tester.test_model('../data/' + str(model_time) + '_model.txt', '../data/' + str(model_time) + '_test_data.txt', edges_of_cell, speed=1.0)


