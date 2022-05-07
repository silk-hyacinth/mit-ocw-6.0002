# Problem Set 4: Simulating the Spread of Disease and Bacteria Population Dynamics
# Name:
# Collaborators (Discussion):
# Time:

import math
import numpy as np
import pylab as pl
import random

random.seed(0)

##########################
# End helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords1, label=y_name1)
    pl.plot(x_coords, y_coords2, label=y_name2)
    pl.legend()
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


##########################
# PROBLEM 1
##########################

class SimpleBacteria(object):

    """A simple bacteria cell with no antibiotic resistance"""

    tag=0

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.tag = SimpleBacteria.tag
        SimpleBacteria.tag += 1

    def __str__(self):
        return str(self.tag)

    def is_killed(self, print_strings=False):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        if random.random() < self.death_prob:
            if print_strings:
                print("bacteria", str(self), "just died")
            return True
        return False

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if random.random() < self.birth_prob * (1 - pop_density):
            return SimpleBacteria(self.birth_prob, self.death_prob)

        raise NoChildException


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop, print_strings=False):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = int(max_pop)
        self.print_strings = print_strings

    def __str__(self):
        tags = [str(bacteria) for bacteria in self.bacteria]
        return str(tags)

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        new_bacteria = [bacteria for bacteria in self.bacteria if not bacteria.is_killed(self.print_strings)]
        new_pop_dens = len(new_bacteria) / self.max_pop

        next_gen = []
        for bacteria in new_bacteria:
            try:
                next_gen.append(bacteria.reproduce(new_pop_dens))
            except NoChildException:
                if self.print_strings:
                    print("bacteria", bacteria, "does not reproduce")
                pass

        new_bacteria += next_gen
        self.bacteria = new_bacteria


# testbacteria = []
# for i in range(100):
#     newbact = SimpleBacteria(random.random(), random.random())
#     testbacteria.append(newbact)
#
# testpatient = Patient(testbacteria, 250, True)
# print(testpatient)
# testpatient.update()
# print(testpatient)


##########################
# PROBLEM 2
##########################

def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    sizes = float(0)
    for population in populations:
        sizes += population[n]
    sizes /= len(populations)
    return sizes


def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials,
                                  num_timesteps=300):
    """
    Run the simulation and plot the graph for problem 2. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Then, plot the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis) You might find the make_one_curve_plot
    function useful.

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    all_populations = []
    for trial_num in range(num_trials):

        trial_bacteria = []
        for i in range(num_bacteria):
            trial_bacteria.append(SimpleBacteria(birth_prob, death_prob))

        trial_patient = Patient(trial_bacteria, max_pop)
        trial_populations = [trial_patient.get_total_pop()]

        for i in range(num_timesteps):
            trial_patient.update()
            trial_populations.append(trial_patient.get_total_pop())

        all_populations.append(trial_populations)
        print("trial", trial_num, "done")

    x_points = []
    y_points = []
    for i in range(num_timesteps):
        x_points.append(i)
        y_points.append(calc_pop_avg(all_populations, i))

    make_one_curve_plot(x_points, y_points, "Number of timesteps", "Population size", "Bacteria growth with birth_prob=" + str(birth_prob) + " death_prob=" + str(death_prob) + " max_pop=" + str(max_pop) + " each trial, modeled with " + str(num_trials) + " trials" )

    # print(x_points, y_points)

    return all_populations


# When you are ready to run the simulation, uncomment the next line
# populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 500, 1000)

##########################
# PROBLEM 3
##########################

def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    p_mean = calc_pop_avg(populations, t)
    total_dist = 0
    for population in populations:
        total_dist += (population[t] - p_mean)**2

    std = (total_dist/len(populations)) ** 0.5

    return std



def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """

    sem = calc_pop_std(populations, t) / (len(populations)**0.5)

    return calc_pop_avg(populations, t), sem*1.96


##########################
# PROBLEM 4
##########################

class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""
    tag=0

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.resistant = resistant
        self.mut_prob = mut_prob
        self.tag = ResistantBacteria.tag
        ResistantBacteria.tag += 1

    def __str__(self):
        return str(self.tag)

    def copy(self):
        return ResistantBacteria(self.birth_prob, self.death_prob, self.resistant, self.mut_prob)

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self, print_strings=False):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.resistant:
            if random.random() < self.death_prob:
                if print_strings:
                    print("bacteria", self, "just died")
                return True

        elif not self.resistant and random.random() < self.death_prob / 4:
            if print_strings:
                print("bacteria", self, "just died")
            return True

        return False

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        if random.random() < self.birth_prob * (1 - pop_density):
            if self.resistant:
                return self.copy()
            else:
                if random.random() < self.mut_prob * (1 - pop_density):
                    return ResistantBacteria(self.birth_prob, self.death_prob, True, self.mut_prob)
                else:
                    return self.copy()
        raise NoChildException


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop, print_strings=False):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        Patient.__init__(self, bacteria, max_pop)
        self.print_strings = print_strings
        self.on_antibiotic = False

    def __str__(self):
        tags = [str(bacteria) + " " + str(bacteria.get_resistant()) for bacteria in self.bacteria]
        return str(self.on_antibiotic) + " " + str(tags)

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True

    def get_resist_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        resist = [bacteria for bacteria in self.bacteria if bacteria.get_resistant()]
        return len(resist)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        new_bacteria = [bacteria for bacteria in self.bacteria if not bacteria.is_killed(self.print_strings)]

        if self.on_antibiotic:
            new_bacteria = [bacteria for bacteria in new_bacteria if bacteria.get_resistant()]

        new_pop_dens = len(new_bacteria) / self.max_pop
        next_gen = []

        for bacteria in new_bacteria:
            try:
                next_gen.append(bacteria.reproduce(new_pop_dens))
            except NoChildException:
                if self.print_strings:
                    print("bacteria", bacteria, "does not reproduce")
                pass

        self.bacteria = new_bacteria + next_gen


# resistantbacteria = []
# for i in range(250):
#     resistantbacteria.append(ResistantBacteria(0.25, 0.25, False, 0.25))
#
# testantibioticpatient = TreatedPatient(resistantbacteria, 500, True)
# print(testantibioticpatient)
# testantibioticpatient.update()
# print(testantibioticpatient)
# testantibioticpatient.set_on_antibiotic()
# testantibioticpatient.update()
# print(testantibioticpatient)

##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials,
                               num_timesteps=(150,250),
                               visualize=False):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute
        num_timesteps (tuple): number of timesteps each simulation will run; first number is before antibiotic and second is after.

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    populations = []
    resistant_pop = []

    for trial in range(num_trials):
        # initiating the variables for each trial
        test_resist_bacteria = []
        for i in range(num_bacteria):
            test_resist_bacteria.append(ResistantBacteria(birth_prob, death_prob, resistant, mut_prob))
        test_antibiotic_patient = TreatedPatient(test_resist_bacteria, max_pop, False)
        trial_populations = [test_antibiotic_patient.get_total_pop()]
        trial_resistant_pop = [test_antibiotic_patient.get_resist_pop()]

        for timestep in range(num_timesteps[0]):
            test_antibiotic_patient.update()
            trial_populations.append(test_antibiotic_patient.get_total_pop())
            trial_resistant_pop.append(test_antibiotic_patient.get_resist_pop())

        test_antibiotic_patient.set_on_antibiotic()

        for timestep in range(num_timesteps[1]):
            test_antibiotic_patient.update()
            # print(test_antibiotic_patient)
            trial_populations.append(test_antibiotic_patient.get_total_pop())
            trial_resistant_pop.append(test_antibiotic_patient.get_resist_pop())

        populations.append(trial_populations)
        resistant_pop.append(trial_resistant_pop)
        # print(populations[0])

    if visualize:
        x_points = []
        y_points_populations_1 = []
        y_points_resistant_pop_2 = []

        for i in range(len(populations[0])):
            x_points.append(i)
            y_points_populations_1.append(calc_pop_avg(populations, i))
            y_points_resistant_pop_2.append(calc_pop_avg(resistant_pop, i))

        make_two_curve_plot(x_points, y_points_populations_1, y_points_resistant_pop_2, "Total population of bacteria", "Resistant population of bacteria", "Number of timesteps", "Number of bacteria",
                            "Bacteria growth with birth_prob=" + str(birth_prob) + " death_prob=" + str(death_prob) + " max_pop=" + str(max_pop) + " each trial, modeled with " + str(num_trials) + " trials")

    return (populations, resistant_pop)



# When you are ready to run the simulations, uncomment the next lines one
# at a time
# total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
#                                                       max_pop=1000,
#                                                       birth_prob=0.3,
#                                                       death_prob=0.2,
#                                                       resistant=False,
#                                                       mut_prob=0.8,
#                                                       num_trials=50,
#                                                       visualize=True)

# total_pop_2, resistant_pop_2 = simulation_with_antibiotic(num_bacteria=100,
#                                                       max_pop=1000,
#                                                       birth_prob=0.17,
#                                                       death_prob=0.2,
#                                                       resistant=False,
#                                                       mut_prob=0.8,
#                                                       num_trials=50,
#                                                       visualize=True)

# print("Simulation 1's average number of total bacteria at time step 30 is", calc_95_ci(total_pop, 30)[0], "with a 95% confidence interval of", calc_95_ci(total_pop, 30)[1])
# print("Simulation 1's average number of total bacteria at time step 299 is", calc_95_ci(total_pop, 299)[0], "with a 95% confidence interval of", calc_95_ci(total_pop, 299)[1])
# print("Simulation 1's average number of resistant bacteria at time step 30 is", calc_95_ci(resistant_pop, 30)[0], "with a 95% confidence interval of", calc_95_ci(resistant_pop, 30)[1])
# print("Simulation 1's average number of resistant bacteria at time step 299 is", calc_95_ci(resistant_pop, 299)[0], "with a 95% confidence interval of", calc_95_ci(resistant_pop, 299)[1])
#
# print("Simulation 2's average number of total bacteria at time step 30 is", calc_95_ci(total_pop_2, 30)[0], "with a 95% confidence interval of", calc_95_ci(total_pop_2, 30)[1])
# print("Simulation 2's average number of total bacteria at time step 299 is", calc_95_ci(total_pop_2, 299)[0], "with a 95% confidence interval of", calc_95_ci(total_pop_2, 299)[1])
# print("Simulation 2's average number of resistant bacteria at time step 30 is", calc_95_ci(resistant_pop_2, 30)[0], "with a 95% confidence interval of", calc_95_ci(resistant_pop_2, 30)[1])
# print("Simulation 2's average number of resistant bacteria at time step 299 is", calc_95_ci(resistant_pop_2, 299)[0], "with a 95% confidence interval of", calc_95_ci(resistant_pop_2, 299)[1])

# this coulda been done with some function but i dont care
