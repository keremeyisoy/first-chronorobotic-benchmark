import numpy as np
import path_finder as pf
import make_video


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
        results = []
        test_data = np.loadtxt(path_data)
        if test_data.ndim != 2:
            return [int(testing_time), 0, 0, 0, 0, 0, 0, 0, 0]

        min_time = np.min(test_data[:, 0])

        number_of_detections = len(test_data[:, 0])
        route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]           # clockwise route
        reverse_route = [(-5, 10), (-7, 1), (2, 3), (-5, 10)]   # counter-clockwise route
        path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
        walls = np.loadtxt(path_borders)
        results.append(int(testing_time))
        results.append(number_of_detections)

        path_finder = pf.PathFinder(path=path_model, edges_of_cell=edges_of_cell)

        '''
        Dummy Model
        '''
        # clockwise
        path_finder.creat_graph(dummy=True)
        path_finder.remove_walls(walls)
        path_finder.find_shortest_path(route=route)
        path_finder.extract_trajectory(min_time, speed=speed)

        results.append(path_finder.extract_interactions(path_data, radius=self.radius_of_robot))

        # counter-clockwise
        path_finder.find_shortest_path(route=reverse_route)
        path_finder.extract_trajectory(min_time, speed=speed)
        results.append(path_finder.extract_interactions(path_data, radius=self.radius_of_robot))


        '''
        Real Model
        '''
        path_finder.creat_graph()
        path_finder.remove_walls(walls)

        # clockwise
        path_finder.find_shortest_path(route=route)
        weight_1 = path_finder.get_mean_path_weight()

        path_finder.extract_trajectory(min_time, speed=speed)
        result_1 = path_finder.extract_interactions(path_data, radius=self.radius_of_robot)
        results.append(result_1)

        # counter-clockwise
        path_finder.find_shortest_path(route=reverse_route)
        weight_2 = path_finder.get_mean_path_weight()
        path_finder.extract_trajectory(min_time, speed=speed)
        result_2 = path_finder.extract_interactions(path_data, radius=self.radius_of_robot)
        results.append(result_2)
        results.append(weight_1)
        results.append(weight_2)

        if weight_1 < weight_2:
            results.append(result_1)
        else:
            results.append(result_2)

        if create_video:
            path_trajectory = '../results/trajectory.txt'
            path_interactions = '../results/interactions.txt'
            vm = make_video.VideoMaker(path_data=path_data, path_borders=path_borders, path_trajectory=path_trajectory, path_interactions=path_interactions)
            vm.make_video(str(model_name) + '/' + str(testing_time), with_robot=True, radius_of_robot=self.radius_of_robot)

        return results


if __name__ == "__main__":

    tester = Tester(radius_of_robot=1.)
    edges_of_cell = np.array([0.5, 0.5])
    print tester.test_model('../models/WHyTeS/1554105948_model.txt', '../data/time_windows/1554105948_test_data.txt', testing_time=1554105948, model_name='WHyTeS', edges_of_cell=edges_of_cell, speed=1.0, create_video=False)
