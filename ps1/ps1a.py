###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time: 2/27

from ps1_partition import get_partitions
import time
import timeit

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # TODO: Your code here
    try:
        file = open(filename, 'r', encoding='utf8')
        cowWeights = {}
        for line in file:
            temp = line.split(",")
            temp[1] = int(temp[1].strip())
            cowWeights[temp[0]] = temp[1]
    except OSError:
        print("file not found")
    except IndexError:
        print("file is wrong format probably")
    finally:
        file.close()

    return cowWeights

# a = load_cows('ps1_cow_data.txt')
# print(a)


# Problem 2
def greedy_cow_transport(cows, limit):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int) DEFAULT WAS 10 but I am changing this to general because wtf, come on
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """


    # TODO: Your code here
    transports = []
    remainingCows = cows.copy()

    # gets rid of all cows above the limit, they are screwed anyways
    for cow in remainingCows:
        if remainingCows[cow] > limit:
            del(remainingCows[cow])

    while len(remainingCows) > 0:
        # print("a", remainingCows)
        transport = []
        left = limit
        bestCow = max(remainingCows, key=cows.get)

        # creates each transport
        while left - cows[bestCow] >= 0:
            transport.append(bestCow)
            left -= remainingCows[bestCow]

            # deletes from remaining
            del(remainingCows[bestCow])
            # print("b", remainingCows, bestCow, left)

            # gets the new best cow to test in the while condition
            try:
                bestCow = max(remainingCows, key=cows.get)
            except ValueError:
                continue

        # adds to final
        transports.append(transport)

    return transports

# yes = greedy_cow_transport(load_cows('ps1_cow_data.txt'), 10)
# print(yes)


# Problem 3
def brute_force_cow_transport(cows,limit):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int) but initially 10 cuz godd amm it you put stuff into the arg
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    workingTransports = []
    cowNameList = cows.keys()

    def checkTransport(transports, limit):
        """

        Args:
            transports: a list of lists, each inner list is each transport
            limit: yes

        Returns: whether or not its valid

        """

        for transport in transports:
            total = 0
            for cow in transport:
                total += cows[cow]

            if total > limit:
                return False

        return True

    for partition in get_partitions(cowNameList):
        if checkTransport(partition, limit):
            workingTransports.append(partition)

    bestTransport = min(workingTransports, key=(lambda k: len(k)))

    return bestTransport



# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    cowList = load_cows('ps1_cow_data.txt')

    # GREEDY ALGO TIME
    greedyTimeStart = timeit.default_timer()

    greedyTrips = greedy_cow_transport(cowList, 10)
    print(greedyTrips)
    print("Greedy algo gives", len(greedyTrips), "trips")

    greedyTimeEnd = timeit.default_timer()
    print(greedyTimeEnd - greedyTimeStart) # difference in time

    # BRUTE FORCE ALGO TIME
    bruteForceTimeStart = timeit.default_timer()

    bruteForceTrips = brute_force_cow_transport(cowList, 10)
    print(bruteForceTrips)
    print("Brute force algo gives", len(bruteForceTrips), "trips")

    bruteForceTimeEnd = timeit.default_timer()
    print(bruteForceTimeEnd - bruteForceTimeStart)


if __name__ == '__main__':
    compare_cow_transport_algorithms()

















# def checkTransport(transports, limit, cows):
#     """
#
#     Args:
#         transport: a list of lists, each inner list is each transport
#         limit: yes
#
#     Returns: whether or not its valid
#
#     """
#
#     for transport in transports:
#         total = 0
#         for cow in transport:
#             total += cows[cow]
#
#         if total > limit:
#             return False
#
#     return True
# yes[1].append("Betsy")
# print(checkTransport(yes, 10, load_cows('ps1_cow_data.txt')))
#
# print("abc")
# print()
#
# no = brute_force_cow_transport(load_cows('ps1_cow_data.txt'), 10)
# print(no)