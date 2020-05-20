import networkx
from networkx.exception import NetworkXError
import numpy as np
import warnings
import pandas as pd
from time import time, clock
from cython_files import interactions









class PathFinder:

    def __init__(self, model_data, edges_of_cell, graph=None):
        self.model = model_data  # formerly path
        self.edges_of_cell = edges_of_cell
        #self.graph = networkx.DiGraph()
        self.graph = graph#.copy()
        self.shortest_path = []
        self.shortest_path_real = None
        self.trajectory = None
        self.x_min = -9.25
        #self.x_max = 2.25#3.0
        #self.y_min = 0.0
        self.y_max = 16.5#16.0
        #self.x_range = self.x_max - self.x_min
        #self.y_range = self.y_max - self.y_min
        #self.shape = (int(self.y_range / edges_of_cell[1]) + 1, int(self.x_range / edges_of_cell[0]) + 1)


    def get_indexes(self, XY):
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


    #"""
    def create_graph(self, path_edges):
        start = time()
        edges = pd.read_csv(path_edges, sep=' ', header=None, engine='c').values
        #edges = np.load(path_edges + str('.npy'))  #!!!
        graph = networkx.DiGraph()
        rounding_error = 0.001
        #angles = self.model[:, 2]
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
        #x = ((self.model[:, 0] - self.x_min)/self.edges_of_cell[0]+1).astype(int)
        #y = ((self.y_max-self.model[:, 1])/self.edges_of_cell[1]+1).astype(int)
        #rows<F6>, columns = self.get_indexes(self.model[:, :2])
        rows, columns = self.get_indexes(edges[:, :2])
        # creating weights and tuples of indices in virtual table
        # y -> rows, x -> columns (reason is visualisation)
        src = list(zip(rows + yr - yf, columns + xr - xf))  # source node
        # weights - diagonal ones should be rised by np.sqrt(2.0)
        neco_pro_test = np.array(list(self.model[:792, 3])*8).reshape(-1)
        #wgt = self.model[:, 3] * (1.0+dg*(np.sqrt(2.0)-1.0))
        wgt = neco_pro_test * (1.0+dg*(np.sqrt(2.0)-1.0))
        #wgt = 1.0+dg*(np.sqrt(2.0)-1.0)
        # create pandas edgelist
        df = pd.DataFrame()
        df[0] = src  # source node
        df[1] = list(zip(rows, columns))  # destination node
        df['weight'] = wgt
        
        finish = time()
        #print('data to nodes: ' + str(finish-start))


        start = time()
        # create graph
        graph = networkx.from_pandas_edgelist(df, 0, 1, 'weight', create_using=networkx.DiGraph)

        finish = time()
        #print('nodes to graph: ' + str(finish-start))
        #print('graph nodes: ' + str(len(graph.nodes)))
        return graph


        

    def remove_walls(self, walls):
        #print('graph nodes before remove: ' + str(len(self.graph.nodes)))
        rows, columns = self.get_indexes(walls)
        #np.save('../data/walls_UTBM.npy', np.unique(zip(rows, columns), axis=0))
        #np.savetxt('../data/walls_UTBM.txt', np.unique(zip(rows, columns), axis=0))
        self.graph.remove_nodes_from(list(zip(rows, columns)))
        # tuple(map(tuple, arr))
        #self.graph.remove_nodes_from(tuple(map(tuple, walls)))
        #print('self graph nodes: ' + str(len(self.graph.nodes)))
        #print('self graph nodes: ' + str(networkx.get_edge_attributes(self.graph,'weight')))
    #"""

        
    def update_graph(self, dummy=False):
        #print('before')
        #for a, b, weight in self.graph.edges(data=True):
        #    #if (b == (24, 17) or b == (23, 17)) and (a == (23, 16) or a == (23, 17)):
        #    #    print(str(a) + str(b) + str(weight['weight']))
        #    print(str(a) + str(b) + str(weight['weight']))
        #print('after')
        #self.create_graph()
        """
        wgt = self.model[:, 3]
        #start = time()
        idx = 0
        count = 0
        #for _, _, weight in self.graph.edges(data=True):
        for a, b, weight in self.graph.edges(data=True):
            weight['weight'] *= wgt[idx]
            count += 1
            if (b == (24, 17) or b == (23, 17)) and (a == (23, 16) or a == (23, 17)):
                print(wgt[idx])
                print(str(a) + str(b) + str(weight['weight']))
            if count == 8:
                count = 0
                idx += 1
        #finish = time()
        #print('update weights: ' + str(finish-start))
        """
        #networkx.set_edge_attributes(self.graph, 'weight', {((a,b), (a+1,b+1)):weight for a, b, weight in self.model[:, :3]})
        #print(self.graph[(1,1)][(1,2)])
        start = time()
        rows, columns = self.get_indexes(self.model[:792, :2])
        finish = time()
        #print('... indexes: ' + str(finish-start))
        start = time()
        if np.shape(self.model)[1]>3:
            zase_neco = np.c_[rows, columns, self.model[:792, 3]]
        else:
            zase_neco = np.c_[rows, columns, self.model[:792, 2]]
        neco = {(a,b):weight for a, b, weight in zase_neco}
        finish = time()
        #print('... dictionary: ' + str(finish-start))
        start = time()
        for a, b, weight in self.graph.edges(data=True):
            weight['weight'] *= neco[b]
            #print(str(a) + str(b) + str(weight['weight']))
        finish = time()
        #print('... update: ' + str(finish-start))




    def extract_trajectory(self, start_time, speed, create_video=False, time_step=0.1):
        directions = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
        self.trajectory = []
        time = start_time
        for i in xrange(1, self.shortest_path_real.shape[0]):
            if self.shortest_path_real[i - 1, 0] == self.shortest_path_real[i, 0] and self.shortest_path_real[i - 1, 1] == self.shortest_path_real[i, 1]:
                continue
            distance = (((self.shortest_path_real[i - 1, 0] - self.shortest_path_real[i, 0]) ** 2) + ((self.shortest_path_real[i - 1, 1] - self.shortest_path_real[i, 1]) ** 2)) ** 0.5
            time_next = time + distance / speed

            x = self.shortest_path_real[i - 1, 0]
            y = self.shortest_path_real[i - 1, 1]
            times = np.arange(time, time_next, time_step)
            # print time_next - time
            x_diff = self.shortest_path_real[i, 0] - x
            y_diff = self.shortest_path_real[i, 1] - y
            # length of step
            x_step = (x_diff)/len(times)
            y_step = (y_diff)/len(times)
            ## angle
            #if x_diff == 0 and y_diff == -0.5:
            #    angle = directions[0]
            #elif x_diff == 0.5 and y_diff == -0.5:
            #    angle = directions[1]
            #elif x_diff == 0.5 and y_diff == 0:
            #    angle = directions[2]
            #elif x_diff == 0.5 and y_diff == 0.5:
            #    angle = directions[3]
            #elif x_diff == 0 and y_diff == 0.5:
            #    angle = directions[4]
            #elif x_diff == -0.5 and y_diff == 0.5:
            #    angle = directions[5]
            #elif x_diff == -0.5 and y_diff == 0:
            #    angle = directions[6]
            #elif x_diff == -0.5 and y_diff == -0.5:
            #    angle = directions[7]
            #else:
            #    print('diffs are different')
            #    print('x_diff: ' + str(x_diff))
            #    print('y_diff: ' + str(y_diff))
            for t in times:
                x = x + x_step
                y = y + y_step
                self.trajectory.append((t, x, y, 2, 2))
                #self.trajectory.append((t, x, y, 2, 2, angle))

            time = time_next
        if create_video:
            np.savetxt('../results/trajectory.txt', np.array(self.trajectory))
        return self.trajectory


    def find_shortest_path(self, route):
        self.shortest_path = []
        rows, columns = self.get_indexes(route)
        route = list(zip(rows, columns))  # destination nodes
        try:
            for i in xrange(1, len(route)):
                src = route[i-1]
                dst = route[i]
                self.shortest_path.extend(list(networkx.dijkstra_path(self.graph, src, dst)))
            # indexes to real coordinates
            #print(self.shortest_path)
            self.shortest_path_real = np.array(self.shortest_path, dtype=np.float64)
            self.shortest_path_real[:,0] = self.y_max - self.edges_of_cell[1]*self.shortest_path_real[:,0] - self.edges_of_cell[1]*0.5
            self.shortest_path_real[:,1] = self.x_min + self.edges_of_cell[0]*self.shortest_path_real[:,1] + self.edges_of_cell[0]*0.5
            self.shortest_path_real = self.shortest_path_real[:,::-1]
        except networkx.NetworkXError:
            print "no path!"


    def extract_interactions(self, data, radius, create_video=False, time_step=0.1):
        radius2 = radius * radius
        #interactions = []
        #intensity = 0  # TODO?

        #interactions = 0
        #counter = 0
        if data.ndim != 2:
            return 0
        """
        for position in self.trajectory:  # numpy version
            tmp = (position[0] - time_step*0.5 < data[:, 0]) & (position[0] + time_step*0.5 > data[:, 0])  # numpy version
            no = np.sum(tmp)  # numpy version

            if no > 0:
                tmp = data[tmp, :]  # numpy version
                #dists = np.sqrt(np.sum((tmp[:, 1:3] - position[1:3])**2, axis=1))
                #tmp = tmp[dists <= radius, :]
                dists = np.sum((tmp[:, 1:3] - position[1:3])**2, axis=1)
                #tmp = tmp[dists <= radius2, :]
                #interactions.append(tmp)
                #tmp = tmp[dists <= radius2, 3]
                #intensity += self._angle_weight(tmp, position[5])
                interactions += np.sum(dists <= radius2)

        for position in self.trajectory:
            for human in data:
                if position[0] - time_step*0.5 < human[0]:
                    if position[0] + time_step*0.5 > human[0]:
                        if np.sum((human[1:3] - position[1:3])**2) <= radius2:
                            interactions += 1

        for position in self.trajectory:
            interactions += np.sum(np.sum((data[(position[0] - time_step*0.5 < data[:, 0]) & (position[0] + time_step*0.5 > data[:, 0]), 1:3] - position[1:3])**2, axis=1) <= radius2)
        """
        #print('length of trajectory: ' + str(len(self.trajectory)))
        encounters = interactions.calculate(np.array(self.trajectory), data, radius2, time_step)
        #print(encounters)
        return encounters
            
        #if interactions != []:
        #    interactions = np.vstack(interactions)
        #
        #if create_video:
        #    np.savetxt('../results/interactions.txt', np.array(interactions))
        #counter = len(interactions)
        #if counter > 0:
        #    #return len(np.unique(np.array(interactions)[:, 4]))
        #    return len(interactions)
        #else:
        #    return 0
        #return intensity

        #return interactions


    def get_mean_path_weight(self):
        total_weight = 0.
        for i in xrange(len(self.shortest_path)-1):
            weight = self.graph.get_edge_data(self.shortest_path[i], self.shortest_path[i+1])
            if weight != None:
                total_weight += weight['weight']
                #if weight['weight'] > total_weight:
                #    total_weight = weight['weight']

        #return total_weight/len(self.shortest_path)
        return total_weight


    def _angle_weight(self, A, b):
        C = A - b
        C = (C + np.pi)%(2.0*np.pi) - np.pi
        C = np.abs(C)/np.pi
        return 2.0*np.sum(C)

