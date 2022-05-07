# 6.0002 Problem Set 5
# Graph optimization
# Name:
# Collaborators:
# Time: 8hr probably at least

#
# Finding shortest paths through MIT buildings
#
import unittest
from graph import Digraph, Node, WeightedEdge

#
# Problem 2: Building up the Campus Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer:
#


# Problem 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """

    newDigraph = Digraph()

    map_file = open(map_filename, 'r')
    for line in map_file:
        parsed = line.split()

        s = Node(parsed[0])
        d = Node(parsed[1])

        if not newDigraph.has_node(s):
            newDigraph.add_node(s)
        if not newDigraph.has_node(d):
            newDigraph.add_node(d)

        newEdge = WeightedEdge(s, d, parsed[2], parsed[3])

        newDigraph.add_edge(newEdge)

    print("Loading map from file...")
    return newDigraph


# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out

filename = 'testmap.txt'
dg = load_map(filename)
print(dg)


#
# Problem 3: Finding the Shortest Path using Optimized Search Method
#
# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
#

# Problem 3b: Implement get_best_path
def get_best_path(digraph, start, end, path, max_dist_outdoors, max_dist_total, best_dist_total,
                  best_path, level=0):
    """
    Finds the shortest path between buildings subject to constraints.

    Parameters:
        max_total_dist:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        path: list composed of [[list of strings], int, int]
            Represents the current path of nodes being traversed. Contains
            a list of node names, total distance traveled, and total
            distance outdoors.
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path
        best_dist: int
            The smallest distance between the original start and end node
            for the initial problem that you are trying to solve
        best_path: list of strings
            The shortest path found so far between the original start
            and end node.

    Returns:
        (list of buildings, total dist, outdoor dist)

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then return None.
    """
    # Objective function: find the shortest distance from start to end within the digraph while staying below a certain
    # outdoor distance

    # unpack path into the local distances
    path[0] = path[0] + [start]
    current_total, current_outdoor = path[1], path[2]

    start_node = digraph.get_node(start)
    end_node = digraph.get_node(end)

    if current_total > max_dist_total or current_outdoor > max_dist_outdoors:
        return None
    elif start_node == end_node:
        return path
    else:
        for next_edge_choice in digraph.get_edges_for_node(start_node):
            next_node = next_edge_choice.get_destination()
            next_total_distance = current_total + int(next_edge_choice.get_total_distance())
            next_outdoor_distance = current_outdoor + int(next_edge_choice.get_outdoor_distance())

            if str(next_node) not in path[0]:
                if best_path is None or next_total_distance <= best_dist_total:
                    next_branch = get_best_path(digraph, str(next_node), end, [path[0], next_total_distance, next_outdoor_distance], max_dist_outdoors, max_dist_total, best_dist_total, best_path, level=level+1)
                    try:
                        best_path = next_branch[0]
                        best_dist_total = next_branch[1]
                    except TypeError:
                        continue
            else:
                print("Already visited!", next_node, path)

    return best_path, best_dist_total,


# d = load_map('mit_map.txt')
# a = d.get_node('32')
# print(a)
# print(d.get_edges_for_node(a))


# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_outdoors):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent outdoors on this path must
    not exceed max_dist_outdoors.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then raises a ValueError.
    """
    # TODO
    path = [[], 0, 0]
    new_best_path = get_best_path(digraph, start, end, path, max_dist_outdoors, max_total_dist, best_dist_total=0,
                                  best_path=None)
    if new_best_path[0] is not None:
        return new_best_path[0]
    else:
        raise ValueError



# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class Ps2Test(unittest.TestCase):
    LARGE_DIST = 99999

    def setUp(self):
        self.graph = load_map("mit_map.txt")

    def test_load_map_basic(self):
        self.assertTrue(isinstance(self.graph, Digraph))
        self.assertEqual(len(self.graph.nodes), 37)
        all_edges = []
        for _, edges in self.graph.edges.items():
            all_edges += edges  # edges must be dict of node -> list of edges
        all_edges = set(all_edges)
        self.assertEqual(len(all_edges), 129)

    def _print_path_description(self, start, end, total_dist, outdoor_dist):
        constraint = ""
        if outdoor_dist != Ps2Test.LARGE_DIST:
            constraint = "without walking more than {}m outdoors".format(
                outdoor_dist)
        if total_dist != Ps2Test.LARGE_DIST:
            if constraint:
                constraint += ' or {}m total'.format(total_dist)
            else:
                constraint = "without walking more than {}m total".format(
                    total_dist)

        print("------------------------")
        print("Shortest path from Building {} to {} {}".format(
            start, end, constraint))

    def _test_path(self,
                   expectedPath,
                   total_dist=LARGE_DIST,
                   outdoor_dist=LARGE_DIST):
        start, end = expectedPath[0], expectedPath[-1]
        self._print_path_description(start, end, total_dist, outdoor_dist)
        dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
        print("Expected: ", expectedPath)
        print("DFS: ", dfsPath)
        self.assertEqual(expectedPath, dfsPath)

    def _test_impossible_path(self,
                              start,
                              end,
                              total_dist=LARGE_DIST,
                              outdoor_dist=LARGE_DIST):
        self._print_path_description(start, end, total_dist, outdoor_dist)
        with self.assertRaises(ValueError):
            directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

    def test_path_one_step(self):
        self._test_path(expectedPath=['32', '56'])

    def test_path_no_outdoors(self):
        self._test_path(
            expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

    def test_path_multi_step(self):
        self._test_path(expectedPath=['2', '3', '7', '9'])

    def test_path_multi_step_no_outdoors(self):
        self._test_path(
            expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

    def test_path_multi_step2(self):
        self._test_path(expectedPath=['1', '4', '12', '32'])

    def test_path_multi_step_no_outdoors2(self):
        self._test_path(
            expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
            outdoor_dist=0)

    def test_impossible_path1(self):
        self._test_impossible_path('8', '50', outdoor_dist=0)

    def test_impossible_path2(self):
        self._test_impossible_path('10', '32', total_dist=100)


if __name__ == "__main__":
    unittest.main()


# # initialize the temp path
#     path[0] += [start]
#
#     if path[2] > max_dist_outdoors: # checks to see if you broke the constraint. if so, then stop this path
#         return None
#     elif not digraph.has_node(digraph.get_node(start)) or not digraph.has_node(digraph.get_node(end)):    # base case: raise error if start and end are broken
#         raise ValueError("Node not in graph")
#     elif start == end:    # base case: start = end: then return and go back up the recursive tree
#         return path
#     else:                   # search: checks all the constraints (are the ones
#         for edge in digraph.get_edges_for_node(digraph.get_node(start)):
#             dest = edge.get_destination()
#             src = edge.get_source()
#             if dest.get_name() not in path: # check for loops
#                 if best_path is None or path[1] < best_dist: # check to make sure we are on the best path
#                     updated_path = path.copy()
#                     updated_path[1] += int(edge.get_total_distance())
#                     updated_path[2] += int(edge.get_outdoor_distance())
#                     new_path = get_best_path(digraph, dest, end, updated_path, max_dist_outdoors, best_dist, best_path)
#
#                     if new_path is not None:
#                         best_path, best_dist = new_path, new_path[1]
#             else:
#                 print(dest, "already in path")
#
#     if best_path is None:
#         print("best_path is None")
#         return None
#     else:
#         return best_path, best_dist








# WORKING CODE BUT UNCLEAN
# # Objective function: find the shortest distance from start to end within the digraph while staying below a certain
#     # outdoor distance
#
#     # unpack path into the local distances
#     path[0] = path[0] + [start]
#     current_total, current_outdoor = path[1], path[2]
#     # print(path, level, 'G')
#
#     start_node = digraph.get_node(start)
#     end_node = digraph.get_node(end)
#     # print(start, end, "A")
#
#     if current_total > max_dist_total or current_outdoor > max_dist_outdoors:
#         return None
#     elif start == end:
#         # print(start, end)
#         return path
#     else:
#         for next_edge_choice in digraph.get_edges_for_node(start_node):
#             # print(start_node, str(digraph.get_edges_for_node(start_node)))
#             next_node = next_edge_choice.get_destination()
#             # print(start, end, next_node, best_dist_out, best_dist_total, "B")
#             next_total_distance = current_total + int(next_edge_choice.get_total_distance())
#             next_outdoor_distance = current_outdoor + int(next_edge_choice.get_outdoor_distance())
#             # print(str(next_node), end, [path[0], next_total_distance, next_outdoor_distance], max_dist_outdoors, max_dist_total, best_dist_total, best_dist_out, best_path, level)
#             # print(next_total_distance, next_outdoor_distance, "eowngiuan")
#             if str(next_node) not in path[0]: #avoid cycling
#                 # print("YTESYSEY")
#                 if best_path is None or next_total_distance <= best_dist_total:
#                     next_branch = get_best_path(digraph, str(next_node), end, [path[0], next_total_distance, next_outdoor_distance], max_dist_outdoors, max_dist_total, best_dist_total, best_path, level=level+1)
#                     # if level == 0:
#                     #     print(next_branch, "2bvf", current_total)
#                     # if next_branch is not None:
#                     #     if next_branch[0] is not None:
#                     try:
#                         best_path = next_branch[0]
#                         best_dist_total = next_branch[1]
#                         # best_dist_out = next_branch[2]
#                     except TypeError:
#                         continue
#
#             else:
#                 print("Already visited!", next_node, path)
#
#     # print(best_path, best_dist_total, best_dist_out, "C")
#     return best_path, best_dist_total, # best_dist_out