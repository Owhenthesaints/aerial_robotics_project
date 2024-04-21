# Examples of basic methods for simulation competition
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# Global variables
on_ground = True
height_desired = 1.0
timer = None
startpos = None
timer_done = None
THRESH_PINK = np.array([[140, 110, 110], [180, 255, 255]])


# All available ground truth measurements can be accessed by calling sensor_data[item], where "item" can take the following values:
# "x_global": Global X position (front positive)
# "y_global": Global Y position (left positive)
# "z_global": Global Z position
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad) 0 is front + right hand rule [-pi;pi]
# "v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "v_forward": Forward velocity (body frame)
# "v_left": Leftward velocity (body frame)
# "v_down": Downward velocity (body frame)
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration
# "range_front": Front range finder distance
# "range_down": Donward range finder distance
# "range_left": Leftward range finder distance 
# "range_back": Backward range finder distance
# "range_right": Rightward range finder distance
# "range_down": Downward range finder distance
# "rate_roll": Roll rate (rad/s)
# "rate_pitch": Pitch rate (rad/s)
# "rate_yaw": Yaw rate (rad/s)


# class implementation of the FSM
class StateEnum(Enum):
    INITIAL_SWEEP = 0
    GO_TO_MIDDLE = 1
    SECOND_SWEEP = 2
    GO_TO_END_LINE = 3
    THIRD_SWEEP = 4
    FIND_LANDING_PAD = 5
    BACK_FIND_PINK = 6
    BACK_TO_PINK = 7
    BACK_TO_START = 8  # do not search LP just go to LP stored in beginning


PINK_FILTER_DEBUG = False
MAP_DEBUG = True
DEFAULT_RESPONSE = (0, 0, height_desired, 0)
LOCAL_AVOIDANCE_LR_THRESH = 0.5
# in pixels
CENTROID_THRESHOLD = 20
YAW_RATE = 1.5
# the threshold to get back to 0
ZERO_THRESH = 0.08
# big area threshold so that we drone is forced to find full square not just side
AREA_THRESHOLD = 500
TURN_LEFT = (0, 0, height_desired, YAW_RATE)
TURN_RIGHT = (0, 0, height_desired, -YAW_RATE)
GO_STRAIGHT = (0.5, 0, height_desired, 0)
GO_BACKWARDS = (-0.5, 0, height_desired, 0)
GO_LEFT = (0, 0.5, height_desired, 0)
GO_RIGHT = (0, -0.5, height_desired, 0)
GO_BACK_RIGHT = (-1, -1, height_desired, 0)
MAP_LENGTH = 5
MAP_WIDTH = 3
MAP_THRESHOLD = 0.1
# halfway point
HALFWAY_LINE = 2.5
END_LINE = 3.5


def divide_map(map):
    """
    divide the map 2s are free squares, 1s are occupied and 0s are unexplored
    """
    map_func = map.copy()
    map_copy = np.where(map > 0, 2, map_func)
    map_copy = np.where(map < 0, 1, map_copy)
    # make sure that I am not modifying other map
    return map_copy.copy()


def make_obstacles_bigger(div_map):
    indices = np.argwhere(div_map == 1)
    for index in indices:
        row, col = index
        if col != 0 and col != len(div_map) - 1:
            div_map[row][col - 1] = 1
            div_map[row][col + 1] = 1
        elif col == 0:
            div_map[row][col + 1] = 1
        elif col == len(div_map) - 1:
            div_map[row][col + 1] = 1
    return div_map


def get_map_scales(shape):
    """
    for some obscure reason map stops at 15
    :return: scale
    """
    return MAP_LENGTH / shape[0], MAP_LENGTH / shape[1]


def get_position_on_map(shape: tuple, x_global: float, y_global: float):
    """
    the objective of this function is to get the equivalent position on map
    shape 0 should be x
    """
    x_scale, y_scale = get_map_scales(shape)
    return int(np.round(x_global / x_scale)), int(np.round(y_global / y_scale))


def make_map_functional(map):
    """
    divide the map then size it
    """
    func_map = map[:, 0:16]
    return divide_map(func_map)


def turn_return(left, state, reversed=False):
    """
    :param left: wether we want to turn left or right
    :param int state: the give state
    :param reversed: wether we want the left parameter to be valid or if we want to turn against given left value
    """
    if not reversed:
        if left:
            return list(TURN_LEFT), state
        return list(TURN_RIGHT), state
    else:
        if left:
            return list(TURN_RIGHT), state
        return list(TURN_LEFT), state


# the point is to have a working map
def initial_sweep(sensor_data, camera_data, map, state):
    # equivalent of static values C++
    def give_attribute(attr: str, value):
        if not hasattr(initial_sweep, attr):
            setattr(initial_sweep, attr, value)

    if sensor_data["y_global"] < 1.5:
        give_attribute("prefered_dir_left", True)
    else:
        give_attribute("prefered_dir_left", False)
    give_attribute("angle_done", False)
    give_attribute("angle_sweep_done", False)

    if initial_sweep.angle_sweep_done:
        del initial_sweep.angle_sweep_done
        del initial_sweep.angle_done
        del initial_sweep.prefered_dir_left
        return list(DEFAULT_RESPONSE), state + 1

    # the objective of this code is to SWEEP an angle to 90 back to 0 in order to feed map
    if not initial_sweep.angle_sweep_done:
        # if gove above 90° do this
        if initial_sweep.angle_done:
            if ((sensor_data["yaw"] > 0 - ZERO_THRESH and not initial_sweep.prefered_dir_left)
                    or (sensor_data["yaw"] < 0 + ZERO_THRESH and initial_sweep.prefered_dir_left)):
                initial_sweep.angle_sweep_done = True
                return list(DEFAULT_RESPONSE), state
            return turn_return(initial_sweep.prefered_dir_left, state, True)
        # if within 90° turn towards prefered_dir
        elif np.pi / 2 > sensor_data["yaw"] > -np.pi / 2:
            return turn_return(initial_sweep.prefered_dir_left, state)
        # if out of [-90;90] angle done
        elif not initial_sweep.angle_done:
            initial_sweep.angle_done = True
            return turn_return(initial_sweep.prefered_dir_left, state, True)
        else:
            raise BrokenPipeError("Not supposed to be here in no_pink condition")

    # trying to turn so that the centroids go to the middle of the screen.

    # go to next state
    del initial_sweep.angle_sweep_done
    del initial_sweep.angle_done
    del initial_sweep.prefered_dir_left
    return list(DEFAULT_RESPONSE), state + 1


def go_to_line(sensor_data, camera_data, map, state, line):
    def give_attribute(attr: str, value):
        if not hasattr(go_to_middle, attr):
            setattr(go_to_middle, attr, value)

    if sensor_data["y_global"] < 1.5:
        give_attribute("preferred_dir_left", True)
    else:
        give_attribute("preferred_dir_left", False)

    if sensor_data["x_global"] > line:
        del go_to_middle.preferred_dir_left
        return list(DEFAULT_RESPONSE), state + 1

    x_index, y_index = get_position_on_map(map.shape, sensor_data["x_global"], sensor_data["y_global"])
    # only keep 15 first collumns
    func_map = make_map_functional(map)
    func_map = make_obstacles_bigger(func_map)
    # create a grid where only obstacles are forbidden
    if go_to_middle.preferred_dir_left:
        if not np.any(func_map[x_index:x_index + 3, y_index] == 1):
            return list(GO_STRAIGHT), state
        elif not np.any(func_map[x_index, y_index:y_index + 2] == 1):
            return list(GO_LEFT), state
        else:
            return list(GO_BACKWARDS), state
    else:
        if not np.any(func_map[x_index:x_index + 3, y_index] == 1):
            return list(GO_STRAIGHT), state
        elif not np.any(func_map[x_index, y_index - 1:y_index + 1] == 1):
            return list(GO_RIGHT), state
        else:
            return list(GO_BACKWARDS), state


def go_to_middle(sensor_data, camera_data, map, state):
    """
    the objective of this function is to get to the middle of the map
    """
    return go_to_line(sensor_data, camera_data, map, state, HALFWAY_LINE)


def go_to_end_line(sensor_data, camera_data, map, state):
    return go_to_line(sensor_data, camera_data, map, state, END_LINE)


def find_landing_pad(sensor_data, camera_data, map, state):
    # preprocess map
    func_map = make_map_functional(map)
    index_x, index_y = get_position_on_map(map.shape, sensor_data["x_global"], sensor_data["y_global"])
    func_map = func_map[index_x:, :]

    # create visit_map and mark everything around the map as visited
    if not hasattr(find_landing_pad, "visit_map"):
        kernel = np.ones((3, 3))
        output_matrix = convolve2d(func_map == 1, kernel, mode='same', boundary='fill', fillvalue=0)
        output_matrix[-1, :] = 1
        output_matrix[:, -1] = 1
        output_matrix[:, 0] = 1
        find_landing_pad.visit_map = output_matrix > 0

    find_landing_pad.visit_map[index_x, index_y] = True


    return list(DEFAULT_RESPONSE), state


def back_find_pink(sensor_data, camera_data, map, state):
    return list(DEFAULT_RESPONSE), state


def back_to_pink(sensor_data, camera_data, map, state):
    return list(DEFAULT_RESPONSE), state


def back_to_start(sensor_data, camera_data, map, state):
    return list(DEFAULT_RESPONSE), state


def default_case(*args):
    raise ValueError("state out of range or not a state", args[3])


FSM_DICO = {
    StateEnum.INITIAL_SWEEP.value: initial_sweep,
    StateEnum.GO_TO_MIDDLE.value: go_to_middle,
    StateEnum.SECOND_SWEEP.value: initial_sweep,
    StateEnum.GO_TO_END_LINE.value: go_to_end_line,
    StateEnum.THIRD_SWEEP.value: initial_sweep,
    StateEnum.FIND_LANDING_PAD.value: find_landing_pad
}


# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    # seting up the static_state value
    map = occupancy_map(sensor_data)

    def access_function(case):
        nonlocal sensor_data, camera_data
        return FSM_DICO.get(case, default_case)(sensor_data, camera_data, map, case)

    if not hasattr(get_command, "static_state"):
        get_command.static_state = StateEnum.INITIAL_SWEEP.value

    # activate the FSM
    control_command, get_command.static_state = access_function(get_command.static_state)
    on_ground = False
    return control_command  # [vx, vy, alt, yaw_rate]


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0  # meter
min_y, max_y = 0, 5.0  # meter
range_max = 2.0  # meter, maximum range of distance sensor
res_pos = 0.2  # meter
conf = 0.2  # certainty given by each measurement
t = 0  # only for plotting

map = np.zeros((int((max_x - min_x) / res_pos), int((max_y - min_y) / res_pos)))  # 0 = unknown, 1 = free, -1 = occupied


def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']

    for j in range(4):  # 4 sensors
        yaw_sensor = yaw + j * np.pi / 2  # yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']

        for i in range(int(range_max / res_pos)):  # range is 2 meters
            dist = i * res_pos
            idx_x = int(np.round((pos_x - min_x + dist * np.cos(yaw_sensor)) / res_pos, 0))
            idx_y = int(np.round((pos_y - min_y + dist * np.sin(yaw_sensor)) / res_pos, 0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break

    map = np.clip(map, -1, 1)  # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map, 1), vmin=-1, vmax=1, cmap='gray',
                   origin='lower')  # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t += 1

    return map


# Control from the exercises
index_current_setpoint = 0


def path_to_setpoint(path, sensor_data, dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
    if on_ground and sensor_data['z_global'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2] - 0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer, 1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], \
        sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm(
        [current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone,
         clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint


def clip_angle(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle
