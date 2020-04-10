import numpy as np
#import path_finder as pf
import path_finder_new as pf
import make_video
#import warnings
from time import time

import pandas as pd
#from pandas.io.parser import CParserError
import pandas.io.common

class Tester:

    def __init__(self, radius_of_robot):
        self.radius_of_robot = radius_of_robot

    def test_model(self, path_model, path_data, testing_time, model_name, edges_of_cell=np.array([0.5, 0.5]), speed=1.0, create_video=False):
        '''

        :param path_model: path for the model output in following format; x y angle weight
        :param path_data: path for testing data
        :param testing_time:
        :param edges_of_cell:
        :param speed: speed of the robot (meters/seconds)
        :param create_video: create a video of trajectory if it is True
        :return: list of values; [testing_time, number_of_detections_in_testing_data, interactions_of_dummy_model_clockwise, interactions_of_dummy_model_counterclockwise, interactions_of_real_model_clockwise, interactions_of_real_model_counterclockwise, total_weight_in_clockwise, total_weight_in_counterclockwise, total_interactions_of_chosen_trajectory]
        '''
        #start = time()
        results = []
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
            #tu = time()
            #test_data = np.loadtxt(path_data)
            #tutaj = time()
            #print('numpy load: ' + str(tutaj-tu))
            #tu = time()
            #test_data = pd.read_csv(path_data, sep=' ', header=None, engine='c', memory_map=True).values#, float_precision='round-trip' returns exatly what numpy
            #tutaj = time()
            #print('pandas load: ' + str(tutaj-tu))
            #print(np.array_equal(test_data,test_data_2))
            #print(np.result_type(test_data))
            #print(np.result_type(test_data_2))
            #print(np.sum(test_data-test_data_2))
            #print(np.sum(test_data==test_data_2))
            #print(np.shape(test_data))
            #print(np.allclose(test_data,test_data_2))
        try:
            #test_data = pd.read_csv(path_data, sep=' ', header=None, engine='c', float_precision='round_trip', usecols=[0, 1, 2, 7]).values#float_precision='round-trip' returns exatly what numpy
            test_data = pd.read_csv(path_data, sep=' ', header=None, engine='c', usecols=[0, 1, 2, 6, 7]).values#float_precision='round-trip' returns exatly what numpy
        #except pd.parser.CParserError:
        except pandas.io.common.EmptyDataError:
            test_data = np.array([])
        
            
        # min_time = np.min(test_data[:, 0])
        min_time = testing_time
        
        if test_data.ndim == 2:
            test_data = test_data[test_data[:,0].argsort(kind='mergesort')]
            number_of_detections = len(np.unique(test_data[:, -1]))
        #elif test_data.ndim == 1 and len(test_data) == 8:
        elif test_data.ndim == 1 and len(test_data) == 5:
            number_of_detections = 1
        else:
            number_of_detections = 0

        #route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]           # clockwise route
        #reverse_route = [(-5, 10), (-7, 1), (2, 3), (-5, 10)]   # counter-clockwise route
        route = np.array([(-5.0, 9.75), (2.0, 2.75), (-7.0, 0.75), (-5.0, 9.75)])           # clockwise route
        reverse_route = np.array([(-5.0, 9.75), (-7.0, 0.75), (2.0, 2.75), (-5.0, 9.75)])   # counter-clockwise route
        path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
        #walls = np.loadtxt(path_borders)
        #walls = pd.read_csv(path_borders, sep=' ', header=None, engine='c', float_precision='round_trip').values#float_precision='round-trip' returns exatly what numpy
        walls = pd.read_csv(path_borders, sep=' ', header=None, engine='c').values#float_precision='round-trip' returns exatly what numpy
        results.append(int(testing_time))
        results.append(number_of_detections)

        #Model = pd.read_csv(path_model, sep=' ', header=None, engine='c', float_precision='round_trip').values
        #Model = pd.read_csv(path_model, sep=' ', header=None, engine='c').values
        if path_model.rsplit('.', 1)[1] == 'txt':
            Model = pd.read_csv(filepath_or_buffer=path_model, sep=' ', header=None, engine='c').values
        elif path_model.rsplit('.', 1)[1] == 'npy':
            Model = np.load(path_model)
        else:
            Model = None
            print('unknown file type')

        """
        Model = np.empty_like(Model0)
        np.copyto(Model, Model0)
        Model0 = np.loadtxt(path_model)
        print(np.array_equal(Model,Model0))
        print(np.result_type(Model))
        print(np.result_type(Model0))
        print(np.sum(Model-Model0))
        print(np.sum(Model==Model0))
        print(np.shape(Model))
        print(np.shape(Model0))
        print(np.allclose(Model,Model0))
        """

        #path_finder = pf.PathFinder(path=path_model, edges_of_cell=edges_of_cell)
        path_finder = pf.PathFinder(model_data=Model, edges_of_cell=edges_of_cell)
        #finish = time()
        #print('first part: ' + str(finish-start))

        '''
        Dummy Model
        '''
        """
        # clockwise
        path_finder.creat_graph(dummy=True)
        path_finder.remove_walls(walls)
        path_finder.find_shortest_path(route=route)
        path_finder.extract_trajectory(min_time, speed=speed)

        results.append(path_finder.extract_interactions(test_data, radius=self.radius_of_robot))

        # counter-clockwise
        path_finder.find_shortest_path(route=reverse_route)
        path_finder.extract_trajectory(min_time, speed=speed)
        results.append(path_finder.extract_interactions(test_data, radius=self.radius_of_robot))
        """


        '''
        Real Model
        '''
        #start_all = time()


        #start = time()
        path_finder.creat_graph()
        #finish = time()
        #print('creat_graph')
        #print(finish-start)

        #start = time()
        path_finder.remove_walls(walls)
        #finish = time()
        #print('remove_walls')
        #print(finish-start)

        #start = time()

        # clockwise
        path_finder.find_shortest_path(route=route)
        #finish = time()
        #print('find_shortest_path')
        #print(finish-start)

        #start = time()
        weight_1 = path_finder.get_mean_path_weight()
        #finish = time()
        #print('get_mean_path_weight')
        #print(finish-start)

        #start = time()

        path_finder.extract_trajectory(min_time, speed=speed, create_video=create_video)
        #finish = time()
        #print('extract_trajectory')
        #print(finish-start)

        #start = time()
        result_1 = path_finder.extract_interactions(test_data, radius=self.radius_of_robot, create_video=create_video)
        results.append(result_1)
        #finish = time()
        #print('extract_interactions')
        #print(finish-start)


        # counter-clockwise
        #start = time()
        path_finder.find_shortest_path(route=reverse_route)
        #finish = time()
        #print('find_shortest_path')
        #print(finish-start)
        #start = time()
        weight_2 = path_finder.get_mean_path_weight()
        #finish = time()
        #print('get_mean_path_weight')
        #print(finish-start)
        #start = time()
        path_finder.extract_trajectory(min_time, speed=speed, create_video=create_video)
        #finish = time()
        #print('extract_trajectory')
        #print(finish-start)
        #start = time()
        result_2 = path_finder.extract_interactions(test_data, radius=self.radius_of_robot, create_video=create_video)
        results.append(result_2)
        #finish = time()
        #print('extract_interactions')
        #print(finish-start)
        results.append(weight_1)
        results.append(weight_2)

        if weight_1 < weight_2:
            results.append(result_1)
        else:
            results.append(result_2)
        
        #finish_all = time()
        #print('second part: ' + str(finish_all-start_all))

        if create_video:
            path_trajectory = '../results/trajectory.txt'
            path_interactions = '../results/interactions.txt'
            vm = make_video.VideoMaker(path_data=path_data, path_borders=path_borders, path_trajectory=path_trajectory, path_interactions=path_interactions)
            vm.make_video(str(model_name) + '/' + str(testing_time), with_robot=True, radius_of_robot=self.radius_of_robot)

        return results


if __name__ == "__main__":

    start = time()
    tester = Tester(radius_of_robot=1.)
    finish = time()
    print('zero part: ' + str(finish-start))
    
    edges_of_cell = np.array([0.5, 0.5])
    start = time()
    print tester.test_model('../models/1_cluster_9_periods/1554105994_model.txt', '../data/time_windows/1554105994_test_data.txt', testing_time=1554105994, model_name='WHyTeS', edges_of_cell=edges_of_cell, speed=1.0, create_video=False)
    finish = time()
    print('first&second part: ' + str(finish-start))
