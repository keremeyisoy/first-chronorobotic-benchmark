"""
this is an example, how to run the method
"""
import directions
import fremen
import tester as tm
import numpy as np
from time import time

# parameters for the method
number_of_clusters = 1
#number_of_spatial_dimensions = 2  # known from data
number_of_spatial_dimensions = 4  # france data
#list_of_periodicities = [21600.0, 43200.0, 86400.0, 86400.0*7.0]  # the most prominent periods, found by FreMEn
#list_of_periodicities = []
#list_of_periodicities = [86400.0]  # the most prominent periods, found by FreMEn
#movement_included = True  # True, if last two columns of dataset are phi and v, i.e., the angle and speed of human.
movement_included = False  # using velocity vector precalculated in dataset.

#structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input


results_file = open("results.txt", "w")
model_times = [1554105948, 1554139930]




TS = np.loadtxt('../data/training_dataset.txt')[:, [0,-1]]
start = time()
W=fremen.build_frequencies(60*60*24, 60*60)
P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_all=True)
finish = time()
#print('time to run fremen: ' + str(finish-start))
#print(P)
number_of_periodicities = 1
#results_file.write(str(number_of_periodicities))


for number_of_clusters in xrange(1,2):
    #print('\n######################\nnumber_of_clusters: ' + str(number_of_clusters))
    #results_file.write("")
    #results_file.write(str(number_of_clusters))
    list_of_periodicities = []

    for no_periodicities in xrange(number_of_periodicities+1):
        #print('\nlist_of_periodicities: ' + str(list_of_periodicities))
        #results_file.write(" ")
        #results_file.write(str(list_of_periodicities))
        structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
        #for i in xrange(4):
        #print(i)
        # load and train the predictor
        start = time()
        dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
        #dirs = dirs.fit('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
        dirs = dirs.fit('../data/training_dataset.txt')
        finish = time()
        #print('time to create model: ' + str(finish-start))
        #results_file.write('\ntime to create model: ' + str(finish-start))
        """
        for i in xrange(number_of_peridocities):
            results_file.write(" ")
            if i > no_periodicities:
                results_file.write("nan")
            else:
                results_file.write(str(list_of_periodicities[i]))
        """
        """
        start = time()
        print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/test_dataset.txt')))
        finish = time()
        print('time to calculate RMSE: ' + str(finish-start))
        """
        X, target = dirs.transform_data('../data/training_dataset.txt')
        pred_for_fremen = dirs.predict(X)
        sample_weights = target - pred_for_fremen
        start = time()
        P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_W=True)
        finish = time()
        #print('time to run fremen: ' + str(finish-start))
        #print P
        #print 1/W
        list_of_periodicities.append(P)
        #print len(W)
        for model_time in model_times:
            start = time()
            #out = dirs.model_to_directions_for_kevin_no_time_dimension()
            #model_time = 1554105948
            #model_time = 1554139930
            results_file.write(" ")
            results_file.write(str(model_time))
            out = dirs.model_to_directions(model_time)
            #np.savetxt('../data/model_of_8angles_0.5m_over_month.txt', out)
            np.savetxt('../data/' + str(model_time) + '_model.txt', out)
            finish = time()
            #print('time to save model for specific time: ' + str(finish-start))
            #print(np.sum(out[:,-1]))
           
            for i in xrange(number_of_periodicities):
                results_file.write(" ")
                if i >= no_periodicities:
                    results_file.write("nan")
                else:
                    results_file.write(str(list_of_periodicities[i]))
 
            results_file.write(str(number_of_periodicities))
            result_file.write(" ")
            results_file.write(" ")
            results_file.write(str(number_of_clusters))


            #tester = tm.Tester()
            tester = tm.Tester(radius_of_robot=1.)

            edges_of_cell = [3600., 0.5, 0.5]
            
            tester_result =  tester.test_model('../data/' + str(model_time) + '_model.txt', '../data/' + str(model_time) + '_test_data.txt', edges_of_cell, speed=1.0)
            for i in range(len(rester_result)):
                results_file.write(" ")
                results_file.write(str(tester_result[i]))
            #print tester_result
            results_file.write("\n")            
    
results_file.close()
