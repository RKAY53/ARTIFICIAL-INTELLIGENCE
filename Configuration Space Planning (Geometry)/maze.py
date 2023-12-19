# maze.py
# Kalaipriyan R
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from state import MazeState, euclidean_distance2
from geometry import does_alien_path_touch_wall, does_alien_touch_wall, is_alien_within_window


class MazeError(Exception):
    pass


class NoStartError(Exception):
    pass


class NoObjectiveError(Exception):
    pass


class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        self.k = k
        self.alien = alien
        self.walls = walls

        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic

        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        if not self.__start:
            # raise SystemExit
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        """"
        Returns True if the given position is the location of an objective
        """
        return waypoint in self.__objective

    def get_start(self):
        assert (isinstance(self.__start, MazeState))
        return self.__start

    def set_start(self, start):
        self.__start = start

    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    # TODO VI
    def filter_valid_waypoints(self):
        valid_waypoints = {i: [] for i in range(len(self.alien.get_shapes()))}

        for shape_idx in range(len(self.alien.get_shapes())):
            for waypoint in self.get_waypoints():
                alien = self.create_new_alien(waypoint[0], waypoint[1], shape_idx)
                if not does_alien_touch_wall(alien, self.walls) and is_alien_within_window(alien, (alien.get_alien_limits()[0][1], alien.get_alien_limits()[1][1])):
                    valid_waypoints[shape_idx].append(waypoint)
        return valid_waypoints
        

    # TODO VI
    def get_nearest_waypoints(self, cur_waypoint, cur_shape):
        distances = [ ]
        waypoints = self.get_valid_waypoints( )[ cur_shape ]
        cur_location = ( cur_waypoint[ 0 ], cur_waypoint[ 1 ], cur_shape )
        for waypoint in waypoints:
            newPoint = ( waypoint[ 0 ], waypoint[ 1 ], cur_shape )
            if cur_waypoint != waypoint and self.is_valid_move( cur_location, newPoint ):
                distances.append((waypoint, euclidean_distance2(cur_waypoint,waypoint)))

        # Sort the distances and return the k nearest waypoints #
        distances = sorted( distances, key = lambda x: x[ 1 ] )
        nearest_neighbors = []
        for i in range( len( distances ) ):
            if i >= self.k:
                break
            nearest_neighbors.append( distances[i][0] )

        return nearest_neighbors
    
    def create_new_alien(self, x, y, shape_idx):
        alien = copy.deepcopy(self.alien)
        alien.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        # print("Shape: ", alien.get_shape())
        return alien

    # TODO VI
    def is_valid_move(self, start, end):
        if (start, end) in self.move_cache:
            return self.move_cache[(start, end)]
        if start[2] != end[2] and (start[0] != end[0] or start[1] != end[1]):
            self.move_cache[(start, end)] = False
            return False
        if start[2] == end[2]:
            return not does_alien_path_touch_wall(self.create_new_alien(start[0], start[1], start[2]), self.walls, (end[0], end[1]))
        else:
            if (start[2] == 0 and end[2]!=1) or (start[2] == 2 and end[2] != 1):
                self.move_cache[(start, end)] = False
                return False
            
            valid_move = not does_alien_touch_wall(self.create_new_alien(start[0], start[1], end[2]), self.walls) and not does_alien_touch_wall(self.create_new_alien(start[0], start[1], start[2]), self.walls)
            self.move_cache[(start, end)] = False
            return valid_move
        

    def get_neighbors(self, x, y, shape_idx):
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        neighbors = [(*end, shape_idx) for end in nearest]
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            if self.is_valid_move(start, end):
                neighbors.append(end)

        return neighbors