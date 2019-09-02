import networkx
from networkx.exception import NetworkXError
import numpy as np
import warnings


class PathFinder:

    def __init__(self, path, edges_of_cell):
        self.model = np.loadtxt(path)
        self.edges_of_cell = edges_of_cell
        self.graph = networkx.DiGraph()
        self.shortest_path = []
        self.shortest_path_real = None
        self.trajectory = None
        self.interactions = []
        self.x_min = -9.25
        self.x_max = 3.0
        self.y_min = 0.0
        self.y_max = 16.0
        # self.x_max = np.max(data[:, 1])
        # self.x_min = np.min(data[:, 1])
        # self.y_max = np.max(data[:, 2])
        # self.y_min = np.min(data[:, 2])
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.shape = (int(self.y_range / edges_of_cell[1]) + 1, int(self.x_range / edges_of_cell[0]) + 1)

    def get_real_coordinate(self, row, column):
        """
        :param row:
        :param column:
        :param x_min:
        :param y_max:
        :param edges_of_cell:
        :return: (x, y) coordiates
        """
        x_length = self.edges_of_cell[0]
        y_length = self.edges_of_cell[1]

        return (self.x_min + x_length*column - x_length*0.5, self.y_max - y_length*row + y_length*0.5)

    def get_index(self, x, y):
        """
        :param x:
        :param y:
        :return: (row, column) index
        """
        x_length = self.edges_of_cell[0]
        y_length = self.edges_of_cell[1]

        return  (int((self.y_max-y)/y_length+1), int((x - self.x_min)/x_length+1))

    def get_weight(self, dst, angle, dummy_model=False):
        if dummy_model:
            return 1.
        
        real_dst = self.get_real_coordinate(dst[0], dst[1])
        k = 0

        while k < len(self.model[:, 0]):
            if self.model[k, 0] == real_dst[0] and self.model[k, 1] == real_dst[1] and self.model[k, 2] == angle:
                return self.model[k, 3]
            k += 1
        return 10000

    def creat_graph(self, dummy=False):
        self.graph = networkx.DiGraph()
        directions = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
        shape = self.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                src = (i, j)
                # if the cell is not on any border
                if i != 0 and j != 0 and i != shape[0]-1 and j != shape[1]-1:
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))

                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif i == 0 and j == shape[1]-1:    # top right corner
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif i == 0 and j == 0:    # top left corner
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

                elif i == shape[0]-1 and j == shape[1]-1:    # bottom right corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))

                elif i == shape[0]-1 and j == 0:    # bottom left corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))

                elif i == 0:    # top border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

                elif i == shape[0]-1:    # bottom border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))

                elif j == shape[1]-1:   # right border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif j == 0:    # left border
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

        return

    def remove_walls(self, walls):

        for position in walls:
            row, column = self.get_index(position[0], position[1])
            try:
                self.graph.remove_node((row, column))
            except NetworkXError:
                pass
        return

    def extract_path(self):
        positions = np.zeros((len(self.shortest_path), 2))

        for i in xrange(len(self.shortest_path)):
            x, y = self.get_real_coordinate(self.shortest_path[i][0], self.shortest_path[i][1])
            positions[i, 0] = x
            positions[i, 1] = y

        np.savetxt('../results/path.txt', positions)
        return positions

    def extract_trajectory(self, start_time, speed):
        self.trajectory = []
        time = start_time

        for i in xrange(1, self.shortest_path_real.shape[0]):
            if self.shortest_path_real[i - 1, 0] == self.shortest_path_real[i, 0] and self.shortest_path_real[i - 1, 1] == self.shortest_path_real[i, 1]:
                continue
            distance = (((self.shortest_path_real[i - 1, 0] - self.shortest_path_real[i, 0]) ** 2) + ((self.shortest_path_real[i - 1, 1] - self.shortest_path_real[i, 1]) ** 2)) ** 0.5
            time_next = time + distance / speed

            x = self.shortest_path_real[i - 1, 0]
            y = self.shortest_path_real[i - 1, 1]
            times = np.arange(time, time_next, 0.1)
            # print time_next - time
            x_step = (self.shortest_path_real[i, 0] - x)/len(times)
            y_step = (self.shortest_path_real[i, 1] - y)/len(times)
            for t in times:
                x = x + x_step
                y = y + y_step
                self.trajectory.append((t, x, y, 2, 2))

            time = time_next

        np.savetxt('../results/trajectory.txt', np.array(self.trajectory))
        return self.trajectory

    def find_shortest_path(self, route):
        self.shortest_path = []
            
        try:
            for i in xrange(1, len(route)):
                src = self.get_index(route[i-1][0], route[i-1][1])
                dst = self.get_index(route[i][0], route[i][1])
                self.shortest_path.extend(list(networkx.dijkstra_path(self.graph, src, dst)))

            self.shortest_path_real = np.zeros((len(self.shortest_path), 2))

            for i in xrange(len(self.shortest_path)):
                x, y = self.get_real_coordinate(self.shortest_path[i][0], self.shortest_path[i][1])
                self.shortest_path_real[i, 0] = x
                self.shortest_path_real[i, 1] = y

        except networkx.NetworkXError:
            print "no path!"

        return self.shortest_path

    def extract_interactions(self, path, radius):
        self.interactions = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.loadtxt(path)
        counter = 0
        if data.ndim != 2:
            return 0
        for position in self.trajectory:
            x_robot = position[1]
            y_robot = position[2]

            indexes = np.where((position[0] - 0.5 < data[:, 0]) & (position[0] + 0.5 > data[:, 0]))

            for index in indexes[0]:
                x_pedestrian = data[index, 1]
                y_pedestrian = data[index, 2]
                dist = ((x_robot - x_pedestrian) ** 2 + (y_robot - y_pedestrian) ** 2) ** 0.5
                if dist <= radius:
                    self.interactions.append((data[index, 0], data[index, 1], data[index, 2], data[index, 7], 3))
                    counter += 1

        np.savetxt('../results/interactions.txt', np.array(self.interactions))

        if counter > 0:
            # return counter
            return len(np.unique(np.array(self.interactions)[:, 3]))
        else:
            return 0

    def extract_centers(self):
        directions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
        k = 0
        centers = np.zeros((self.shape[0]*self.shape[1]*8, 3))
        for j in xrange(self.shape[1]):
            for i in xrange(self.shape[0]):
                x, y = self.get_real_coordinate(i, j)
                for z in xrange(8):
                    centers[k, 0] = x
                    centers[k, 1] = y
                    centers[k, 2] = directions[z]
                    k+=1

        np.savetxt('../results/centers.txt', centers)

    def get_path_weight(self):
        total_weight = 0
        for i in xrange(len(self.shortest_path)-1):
            weight = self.graph.get_edge_data(self.shortest_path[i], self.shortest_path[i+1])
            if weight != None:
                total_weight += weight['weight']

        return total_weight

    def get_mean_path_weight(self):
        total_weight = 0.
        for i in xrange(len(self.shortest_path)-1):
            weight = self.graph.get_edge_data(self.shortest_path[i], self.shortest_path[i+1])
            if weight != None:
                total_weight += weight['weight']

        return total_weight/len(self.shortest_path)


if __name__ == "__main__":

    edges_of_cell = np.array([0.5, 0.5])
    path_finder = PathFinder('../models/WHyTeS/1554105948_model.txt', edges_of_cell)
    path_finder.creat_graph()
    walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
    path_finder.remove_walls(walls)

    route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]
    path_finder.find_shortest_path(route)
    path_finder.extract_path()
    path = np.loadtxt('../results/path.txt')
    testing_data_path = '../data/time_windows/1554105948_test_data.txt'


    data = np.loadtxt(testing_data_path)
    path_finder.extract_trajectory(np.min(data[:, 0]), speed=1)
    print path_finder.get_path_weight()
    print path_finder.extract_interactions(testing_data_path, 2.)
