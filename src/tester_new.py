import numpy as np
#import path_finder as pf
import path_finder_new as pf
import make_video
#import warnings
from time import time
import networkx


import pandas as pd
#from pandas.io.parser import CParserError
import pandas.io.common

class Tester:

    def __init__(self):
        pass


    def create_scenario(self, radius=1.0, speed=1.0, time_step=0.1,
                              path_edges='../data/graph_basic.npy', 
                              path_walls='../data/artificial_boarders_of_space_in_UTBM.npy', 
                              route = np.array([(-5.0, 9.75), (-7.0, 0.75), (2.0, 2.75), (-5.0, 9.75)]),
                              edges_of_cell=np.array([0.5, 0.5]), 
                              x_min=-9.25, y_max=16.5):

        self.edges_of_cell = edges_of_cell
        self.x_min = x_min
        self.y_max = y_max
        self.radius_of_robot = radius
        self.robot_speed = speed
        self.time_step = time_step
        self.route = route
        self.graph, self.def_edgs = self._create_graph(path_edges, path_walls)
        self.graph_copy = self.graph.copy()  # for testing
        return self


    def _remove_weights(self):
        for a, b, weight in self.graph.edges(data=True):
            weight['weight'] = self.def_edgs[(a, b)]


    def _create_graph(self, path_edges, path_walls):
        edges = np.load(path_edges)
        walls = np.load(path_walls)
        graph = networkx.DiGraph()
        rounding_error = 0.001
        angles = edges[:, 2]
        # for every x, y, angle we have a weight
        # x, y here is destination node
        # from angle, we are looking for source node
        # assuming 8 possible directions from source to destination node 
        xr = (angles > rounding_error-np.pi/2.0) & (angles < -rounding_error+np.pi/2.0)  # x rises
        xf = (angles < -rounding_error-np.pi/2.0) | (angles > rounding_error+np.pi/2.0)  # x falls
        yr = (angles > rounding_error+0.0) & (angles < -rounding_error+np.pi)  # y rises
        yf = (angles < -rounding_error+0.0) & (angles > rounding_error-np.pi)  # y falls
        dg = (xr & yr) | (xr & yf) | (xf & yr) | (xf & yf)  # diagonal edge
        # changing real coordinates to indexes (becouse of dijkstra)
        rows, columns = self._get_indexes(edges[:, :2])
        # destination node
        dst = list(zip(rows, columns))
        # creating weights and tuples of indices in virtual table
        # y -> rows, x -> columns (reason is visualisation)
        src = list(zip(rows + yr - yf, columns + xr - xf))  # source node
        # weights - diagonal ones should be rised by np.sqrt(2.0)
        wgt = 1.0+dg*(np.sqrt(2.0)-1.0)
        # create pandas edgelist
        df = pd.DataFrame()
        df[0] = src  # source node
        df[1] = dst  # destination node
        df['weight'] = wgt
        # create dictionary as a default graph values
        def_edgs = {(src, dst):wgt for src, dst, wgt in df.values}
        # create graph
        graph = networkx.from_pandas_edgelist(df, 0, 1, 'weight', create_using=networkx.DiGraph)
        #print('graph nodes: ' + str(len(graph.nodes)))
        # remove walls and obstacles
        rows, columns = self._get_indexes(walls)
        graph.remove_nodes_from(list(zip(rows, columns)))
        #print('graph nodes: ' + str(len(graph.nodes)))
        return graph, def_edgs


    def _get_indexes(self, XY):
        """
        objective:
            to transform real coordinates to indexes in the virtual table
        input:
            XY ... np.array 2xn, spatial coordinates (X, Y) of measurements
        output:
            rows ... np.array 1xn, Y transformed into row index
            columns ... np.array 1xn, X transformed into column index
        """
        columns = ((XY[:, 0] - self.x_min)/self.edges_of_cell[0]+1).astype(int)
        rows = ((self.y_max-XY[:, 1])/self.edges_of_cell[1]+1).astype(int)

        return rows, columns




    def test_model(self, path_model, path_data, testing_time, model_name):
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
        """
        if path_data.rsplit('.', 1)[1] == 'txt':
            try:
                #test_data = pd.read_csv(path_data, sep=' ', header=None, engine='c', usecols=[0, 1, 2, 6, 7]).values
                test_data = pd.read_csv(path_data, sep=' ', header=None, engine='c', usecols=[0, 1, 2]).values.reshape(-1, 3)
                #print(path_data.rsplit('.', 1))
                np.save(path_data.rsplit('.', 1)[0] + '.npy', test_data.reshape(-1, 3)) #!!! bacha, pro test! :)
                #test_data = np.load(path_data) #!!! bacha, pro test! :)
            except:
                test_data = np.array([]).reshape(-1, 3)
                np.save(path_data.split('.')[0] + '.npy', test_data) #!!! bacha, pro test! :)
        
            print(test_data)
            if test_data.ndim == 2:
                number_of_detections = len(test_data)
            #elif test_data.ndim == 1 and len(test_data) == 5:
            elif test_data.ndim == 1 and len(test_data) == 3:
                number_of_detections = 1
            else:
                number_of_detections = 0
        
        elif path_data.rsplit('.', 1)[1] == 'npy':
            test_data = np.load(path_model)
            number_of_detections = len(test_data)
        else:
            test_data = None
            print('unknown file type')
        """

        test_data = np.load(path_data) #!!! bacha, musi se jeste vyrobit :)
        number_of_detections = len(test_data)

        results.append(int(testing_time))
        results.append(number_of_detections)

        """
        if path_model.rsplit('.', 1)[1] == 'txt':
            Model = pd.read_csv(filepath_or_buffer=path_model, sep=' ', header=None, engine='c').values
            print(path_model.rsplit('.', 1))
            np.save(path_model.rsplit('.', 1)[0] + '.npy', Model) #!!! bacha, pro test! :)
        elif path_model.rsplit('.', 1)[1] == 'npy':
            Model = np.load(path_model)
        else:
            Model = None
            print('unknown file type')
        """
        Model = np.load(path_model) #!!! bacha, pro test! :)
        #finish = time()
        #print('loading part: ' + str(finish-start))


        #start = time()
        path_finder = pf.PathFinder(model_data=Model, edges_of_cell=self.edges_of_cell, graph=self.graph)#.copy())
        #finish = time()
        #print('create_object')
        #print(finish-start)



        '''
        Real Model
        '''


        #start = time()
        path_finder.update_graph()
        #finish = time()
        #print('update_graph')
        #print(finish-start)

        # clockwise
        #start = time()
        path_finder.find_shortest_path(route=self.route)
        #finish = time()
        #print('find_shortest_path')
        #print(finish-start)

        #start = time()
        weight = path_finder.get_mean_path_weight()
        #finish = time()
        #print('get_mean_path_weight')
        #print(finish-start)

        #start = time()
        path_finder.extract_trajectory(testing_time, speed=self.robot_speed, time_step=self.time_step)
        #finish = time()
        #print('extract_trajectory')
        #print(finish-start)

        #start = time()
        encounters = path_finder.extract_interactions(test_data, radius=self.radius_of_robot, time_step=self.time_step)
        #finish = time()
        #print('extract_interactions')
        #print(finish-start)

        results.append(encounters)
        results.append(weight)

        #print('original graph changed?')
        #print(cmp(networkx.get_edge_attributes(self.graph,'weight'), networkx.get_edge_attributes(self.graph_copy,'weight')))
        #start = time()
        self._remove_weights()
        #finish = time()
        #print('remove weights')
        #print(finish-start)
        #print('original graph changed?')
        #print(cmp(networkx.get_edge_attributes(self.graph,'weight'), networkx.get_edge_attributes(self.graph_copy,'weight')))

        return results


if __name__ == "__main__":

    start = time()
    tester = Tester().create_scenario()
    finish = time()
    print('zero part: ' + str(finish-start))
    
    start = time()
    #out = tester.test_model('../models/1_cluster_9_periods/1554105994_model.npy', '../data/time_windows/1554105994_test_data.npy', testing_time=1554105994, model_name='WHyTeS')
    out = tester.test_model('../models/model_gmm_fremen_c_5_p_5/1554105994_model.npy', '../data/time_windows/1554105994_test_data.npy', testing_time=1554105994, model_name='WHyTeS')
    finish = time()
    print('first&second part: ' + str(finish-start))
    print(out)
