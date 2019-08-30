import numpy as np
import cv2
from datetime import datetime


class VideoMaker:

    def __init__(self, path_data, path_borders, size_factor=40, path_trajectory=None, path_interactions=None):
        data = np.loadtxt(path_data)
        data = data[:, [0,1,2,7]]
        self.data = np.c_[data, np.ones(data[:, 0].shape)]  #t x y id label
        self.intersections = None
        if path_interactions is not None:
            self.intersections = np.loadtxt(path_interactions)

        if path_trajectory is not None and path_interactions is not None:
            self.trajectory = np.loadtxt(path_trajectory)
            if self.trajectory.shape[1] == 1:
                self.intersections = self.intersections.reshape(1, self.data.shape[1])
        self.path_borders = path_borders
        # self.x_max = np.max(self.data[:, 1])
        # self.x_min = np.min(self.data[:, 1])
        # self.y_max = np.max(self.data[:, 2])
        # self.y_min = np.min(self.data[:, 2])
        self.x_min = -9.25
        self.x_max = 3.0
        self.y_min = 0.0
        self.y_max = 16.0
        if path_trajectory is not None:
            self.time_max = int(max(np.max(data[:, 0]), np.max(self.trajectory[:, 0])))
            self.time_min = int(min(np.min(data[:, 0]), np.min(self.trajectory[:, 0])))
        else:
            self.time_max = int(np.max(data[:, 0]))
            self.time_min = int(np.min(data[:, 0]))
        self.size_factor = size_factor
        self.shape = (int((self.y_max - self.y_min) * size_factor),
                      int((self.x_max - self.x_min) * size_factor))

    def _draw_grid(self, frame, line_color=(0, 255, 0), thickness=1, type=cv2.LINE_AA):

        for i in xrange(int(self.x_min), int(self.x_max), 1):
            x, y = self.get_frame_index(i, self.y_min)
            cv2.line(frame, (x, 0), (x, y), color=line_color, lineType=type, thickness=thickness)

        for i in xrange(int(self.y_min), int(self.y_max), 1):
            x, y = self.get_frame_index(self.x_max, i)
            cv2.line(frame, (x, y), (0, y), color=line_color, lineType=type, thickness=thickness)

        return

    def get_frame_index(self, x, y):

        return (int((x - self.x_min) * self.size_factor), int((self.y_max - y) * self.size_factor))

    def _create_empty_frame(self):
        frame = cv2.cvtColor(np.full(self.shape, 255, np.uint8), cv2.COLOR_GRAY2BGR)
        self._draw_grid(frame, (150, 150, 150), 1, cv2.LINE_4)
        borders = np.loadtxt(self.path_borders)
        for i in xrange(len(borders[:, 0])):
            pos = self.get_frame_index(borders[i, 0], borders[i, 1])
            cv2.circle(frame, pos, int(0.1 * self.size_factor),
                       (0, 0, 0), int(0.05 * self.size_factor))

        return frame

    def make_video (self, output_name, fps=20, with_robot=True, radius_of_robot=0.5):
        if with_robot:
            if self.intersections is not None and len(self.intersections) > 0:
                concatenated = np.concatenate((self.data, self.trajectory, self.intersections), axis=0)
            else:
                concatenated = np.concatenate((self.data, self.trajectory), axis=0)
        else:
            concatenated = self.data
        concatenated = concatenated[concatenated[:, 0].argsort()]

        times = np.arange(self.time_min, self.time_max, 0.1)
        counter = 0
        k = 0

        output_file = '../results/%s.avi' % str(output_name)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(output_file, fourcc, fps, (self.shape[1], self.shape[0]))

        pedestrians = []
        empty_frame = self._create_empty_frame()
        last_pos_robot = (0, 0)
        write_flag = False
        for time in times:
            frame = empty_frame.copy()
            name = str(time)
            del pedestrians[:]

            while k < len(concatenated[:, 0]) and int(time*10) == int(concatenated[k, 0]*10):
                write_flag = True
                # labeling the robot
                if int(concatenated[k, 4]) == 2 and with_robot:
                    last_pos_robot = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])

                # labeling the pedestrians
                if int(concatenated[k, 4]) == 1:
                    pos = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])
                    cv2.circle(frame, pos, int(0.1 * self.size_factor),
                               (250, 0, 0), int(0.1 * self.size_factor))
                    cv2.putText(frame, str(int(concatenated[k, 3])), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)

                # labeling the intersections
                if int(concatenated[k, 4]) == 3:
                    pos = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])
                    cv2.circle(frame, pos, int(1 * self.size_factor),
                               (0, 0, 255-concatenated[k, 3] * 500), int(0.1 * self.size_factor))
                    cv2.putText(frame, 'HIT!!', (pos[0], pos[1]-1*self.size_factor), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                lineType=cv2.LINE_4)

                k += 1

            if with_robot:
                # Radius of robot
                cv2.circle(frame, last_pos_robot, int(radius_of_robot * self.size_factor),
                           (0, 250, 0), int(0.05 * self.size_factor))
                # Center of robot
                cv2.circle(frame, last_pos_robot, int(0.1 * self.size_factor),
                           (0, 250, 0), int(0.1 * self.size_factor))
                cv2.putText(frame, 'Robot', (int(last_pos_robot[0]-radius_of_robot/2 * self.size_factor), last_pos_robot[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                            lineType=cv2.LINE_4)

            cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)
            cv2.putText(frame, datetime.utcfromtimestamp(time).strftime('Day:%d, Time:%H:%M:%S'), (int(0.1*self.size_factor), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.01*self.size_factor, (0, 0, 255), lineType=cv2.LINE_4)
            counter += 1
            if write_flag:
                out.write(frame)
                write_flag = False

        cv2.destroyAllWindows()
        print 'saving video ' + str(output_file)
        out.release()


if __name__ == "__main__":

    path_data = '../data/time_windows/1554105948_test_data.txt'
    path_trajectory = '../results/trajectory.txt'
    path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
    path_interactions = '../results/intersections.txt'
    vm = VideoMaker(path_data, path_borders, size_factor=40, path_interactions=path_interactions, path_trajectory=path_trajectory)

    vm.make_video(output_name='test', with_robot=True, fps=10, radius_of_robot=2.)

    # times = np.loadtxt('../data/test_times.txt')
    # path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
    # for time in times:
    #     path_data = '../data/time_windows/for_video/' + str(int(time)) + '_test_data.txt'
    #
    #     vm = VideoMaker(path_data, path_borders)
    #     vm.make_video(output_name=int(time), with_robot=False, fps=10)

