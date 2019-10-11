import numpy as np
import path_finder as pf
import make_video
import warnings
from time import time


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
        start = time()
        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_data = np.loadtxt(path_data)
        # min_time = np.min(test_data[:, 0])
        min_time = testing_time
        
        if test_data.ndim == 2:
            number_of_detections = len(np.unique(test_data[:, 7]))
        elif test_data.ndim == 1 and len(test_data) == 8:
            number_of_detections = 1
        else:
            number_of_detections = 0

        route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]           # clockwise route
        reverse_route = [(-5, 10), (-7, 1), (2, 3), (-5, 10)]   # counter-clockwise route
        path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
        walls = np.loadtxt(path_borders)
        results.append(int(testing_time))
        results.append(number_of_detections)

        path_finder = pf.PathFinder(path=path_model, edges_of_cell=edges_of_cell)
        finish = time()
        print('first part: ' + str(finish-start))

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
        start = time()


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

        path_finder.extract_trajectory(min_time, speed=speed)
        #finish = time()
        #print('extract_trajectory')
        #print(finish-start)

        #start = time()
        result_1 = path_finder.extract_interactions(test_data, radius=self.radius_of_robot)
        results.append(result_1)
        #finish = time()
        #print('extract_interactions')
        #print(finish-start)


        # counter-clockwise
        path_finder.find_shortest_path(route=reverse_route)
        weight_2 = path_finder.get_mean_path_weight()
        path_finder.extract_trajectory(min_time, speed=speed)
        result_2 = path_finder.extract_interactions(test_data, radius=self.radius_of_robot)
        results.append(result_2)
        results.append(weight_1)
        results.append(weight_2)

        if weight_1 < weight_2:
            results.append(result_1)
        else:
            results.append(result_2)
        
        finish = time()
        print('second part: ' + str(finish-start))

        if create_video:
            path_trajectory = '../results/trajectory.txt'
            path_interactions = '../results/interactions.txt'
            vm = make_video.VideoMaker(path_data=path_data, path_borders=path_borders, path_trajectory=path_trajectory, path_interactions=path_interactions)
            vm.make_video(str(model_name) + '/' + str(testing_time), with_robot=True, radius_of_robot=self.radius_of_robot)

        return results


if __name__ == "__main__":

    tester = Tester(radius_of_robot=1.)
    edges_of_cell = np.array([0.5, 0.5])
    print tester.test_model('../models/1_cluster_9_periods/1554105994_model.txt', '../data/time_windows/1554105994_test_data.txt', testing_time=1554105994, model_name='WHyTeS', edges_of_cell=edges_of_cell, speed=1.0, create_video=False)
