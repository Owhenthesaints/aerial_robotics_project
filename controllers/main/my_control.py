# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from enum import Enum
from scipy.ndimage import measurements

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
    SEARCH_PINK = 2
    GO_TO_PINK = 3
    GO_TO_FZ = 4
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
YAW_RATE = 1
# big area threshold so that we drone is forced to find full square not just side
AREA_THRESHOLD = 500
TURN_LEFT = (0, 0, height_desired, YAW_RATE)
TURN_RIGHT = (0, 0, height_desired, -YAW_RATE)
GO_STRAIGHT = (0.5, 0, height_desired, 0)
GO_BACKWARDS = (-0.5, 0, height_desired, 0)
GO_LEFT = (0, 1, height_desired, 0)
GO_RIGHT = (0, 1, height_desired, 0)
MAP_LENGTH = 5
MAP_WIDTH = 3
MAP_THRESHOLD = 0.1


def divide_map(map):
    """
    divide the map 2s are free squares, 1s are occupied and 0s are unexplored
    """
    map_copy = np.where(map > 0, 2, map)
    map_copy = np.where(map < 0, 1, map_copy)
    return map_copy

def get_map_scales(map):
    """
    :return: scale
    """
    return MAP_LENGTH/map.shape[0], MAP_WIDTH/map.shape[1]


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
def initial_sweep(sensor_data, camera_data, map):
    # init state and if done with state return state+1
    state = StateEnum.INITIAL_SWEEP.value

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
        return list(DEFAULT_RESPONSE), state + 1

    # the objective of this code is to SWEEP an angle to 90 back to 0 in order to feed map
    if not initial_sweep.angle_sweep_done:
        # if gove above 90° do this
        if initial_sweep.angle_done:
            if ((sensor_data["yaw"] > 0 and not initial_sweep.prefered_dir_left)
                    or (sensor_data["yaw"] < 0 and initial_sweep.prefered_dir_left)):
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
    return list(DEFAULT_RESPONSE), state + 1


def go_to_middle(sensor_data, camera_data, map):
    """
    the objective of this function is to get to the middle of the map
    """
    state = StateEnum.GO_TO_MIDDLE.value
    return list(DEFAULT_RESPONSE), state

def search_pink(sensor_data, camera_data, map):
    state = StateEnum.SEARCH_PINK.value
    useable_map = divide_map(map)
    scale_x, scale_y = get_map_scales(map)

    def give_attribute(attr: str, value):
        if not hasattr(search_pink, attr):
            setattr(search_pink, attr, value)

    give_attribute("line_objective", None)
    if sensor_data["y_global"] < 1.5:
        give_attribute("prefered_dir_left", True)
    else:
        give_attribute("prefered_dir_left", False)
    give_attribute("optimal_line", False)

    if MAP_DEBUG:
        cv2.imshow("map", map)
        cv2.waitKey(1)

    hsv = cv2.cvtColor(camera_data, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only pink colors
    square_mask = cv2.inRange(hsv, THRESH_PINK[0], THRESH_PINK[1])

    # defining them outer scope
    stats = largest_component_label = largest_component_mask = None

    # if pink found in image find biggest component
    if np.any(square_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(square_mask)
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_mask = np.uint8(labels == largest_component_label)
        if PINK_FILTER_DEBUG and largest_component_mask is not None:
            cv2.imshow('Camera Feed', largest_component_mask * 255)
            cv2.waitKey(1)

    # check whether pink square has been found
    def no_pink():
        nonlocal square_mask, stats, largest_component_label
        return not np.any(square_mask) or stats[largest_component_label, cv2.CC_STAT_AREA] <= AREA_THRESHOLD

    if no_pink():
        if search_pink.optimal_line:
            if search_pink.prefered_dir_left:
                return list(GO_LEFT), state
            else:
                return list(GO_RIGHT), state
        # find row with most freedom of movement
        elif search_pink.line_objective is None:
            largest_row = 0
            largest_row_count = 0
            for index, row in enumerate(useable_map):
                # disallow 0
                if index == 0:
                    continue
                count = np.sum(row==2)
                if count>largest_row_count:
                    largest_row_count = count
                    largest_row = index
            search_pink.line_objective = largest_row
        elif search_pink.line_objective is not None:
            if search_pink.line_objective*scale_x > sensor_data["x_global"] - MAP_THRESHOLD:
                return list(GO_STRAIGHT), state
            elif search_pink.line_objective*scale_x < sensor_data["x_global"]+MAP_THRESHOLD:
                return list(GO_BACKWARDS), state
            else:
                search_pink.optimal_line = True




        #go towards that row
    else:
        # from centroid trying to get pink square in center
        centroid = measurements.center_of_mass(largest_component_mask)

        if len(camera_data)/2 +1- CENTROID_THRESHOLD > centroid[1] > len(camera_data) /2 +1 + CENTROID_THRESHOLD:
            search_pink.camera_sweeping_done = True
        # if centroid to the right yaw right else yaw left until centroid in middle of screen
        if centroid[1] > len(camera_data) / 2 + 1 + CENTROID_THRESHOLD:
            return list(TURN_RIGHT), state
        elif centroid[1] < len(camera_data) / 2 + 1 - CENTROID_THRESHOLD:
            return list(TURN_LEFT), state

    return list(DEFAULT_RESPONSE), state


def go_to_pink(sensor_data, camera_data, map):
    state = StateEnum.GO_TO_PINK.value
    return list(GO_STRAIGHT), state


def go_to_fz(sensor_data, camera_data, map):
    case = StateEnum.GO_TO_FZ
    return list(DEFAULT_RESPONSE), case


def find_landing_pad(sensor_data, camera_data, map):
    case = StateEnum.FIND_LANDING_PAD
    return list(DEFAULT_RESPONSE), case


def back_find_pink(sensor_data, camera_data, map):
    case = StateEnum.BACK_FIND_PINK
    return list(DEFAULT_RESPONSE), case


def back_to_pink(sensor_data, camera_data, map):
    case = StateEnum.BACK_TO_PINK
    return list(DEFAULT_RESPONSE), case


def back_to_start(sensor_data, camera_data, map):
    case = StateEnum.BACK_TO_START
    return list(DEFAULT_RESPONSE), case


def default_case(*args):
    raise ValueError("state out of range")


FSM_DICO = {
    StateEnum.INITIAL_SWEEP.value: initial_sweep,
    StateEnum.GO_TO_MIDDLE.value: go_to_middle,
    StateEnum.SEARCH_PINK.value: search_pink,
    StateEnum.GO_TO_PINK.value: go_to_pink,
    StateEnum.GO_TO_FZ.value: go_to_fz,
    StateEnum.FIND_LANDING_PAD.value: find_landing_pad,
    StateEnum.BACK_FIND_PINK.value: back_find_pink,
    StateEnum.BACK_TO_PINK.value: back_to_pink,
    StateEnum.BACK_TO_START.value: back_to_start
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
        return FSM_DICO.get(case, default_case)(sensor_data, camera_data, map)

    if not hasattr(get_command, "static_state"):
        get_command.static_state = StateEnum.INITIAL_SWEEP.value

    # activate the FSM
    control_command, get_command.static_state = access_function(get_command.static_state)

    on_ground = False

    print(sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"])
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
