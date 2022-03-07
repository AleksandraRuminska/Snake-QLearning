from enum import IntEnum


class ObjectColor:
    """ Object color used on observation encoding """
    empty = 0, 0, 0
    apple = 255, 0, 0
    own_head = 0, 0, 255
    own_body = 0, 255, 0
    other_head = 128, 128, 255
    other_body = 128, 255, 128
    dead_head = 128, 128, 128
    dead_body = 64, 64, 64


class GridType(IntEnum):
    """ Style of grid to use for environment """
    square = 0
    hex = 1


class Action4(IntEnum):
    """ Actions to be taken by an 
    agent in a square grid """
    forward = 0
    right = 1
    down = 3
    left = 2


class Direction4(IntEnum):
    """ Square grid orientations """
    north = 0
    east = 1
    south = 2
    west = 3

    def add_action(self, action):
        if action == Action4.left:
            return Direction4.west

        if action == Action4.right:
            return Direction4.east

        if action == Action4.forward:
            return Direction4.north

        if action == Action4.down:
            return Direction4.south


    def add_to_point(self, point):
        if self == Direction4.north:
            return point[0], point[1] - 1
        if self == Direction4.east:
            return point[0] + 1, point[1]
        if self == Direction4.south:
            return point[0], point[1] + 1
        return point[0] - 1, point[1]

