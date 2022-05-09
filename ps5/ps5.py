# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: silk-hyacinth
# Collaborators (discussion):
# Time: early May 2022, lots of hours because this pset was so long

import pylab
import numpy
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_city_names(self):
        return list(self.rawdata.keys())

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # TODO
    final_arrays = []

    for deg in degs:
        final_arrays.append(pylab.polyfit(x, y, deg)) # appends the fitted model to final_arrays

    # print(final_arrays)
    return final_arrays

    # return [pylab.polyfit(x, y, deg) for deg in degs]  # LOL list comprehension. it does what the docstring says lol..


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    predictionError = ((y-estimated)**2).sum()
    meanError = ((y-numpy.mean(y))**2).sum()
    return 1-(predictionError/meanError)

def evaluate_models_on_training(x, y, models, titlestr=""):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # TODO
    pylab.figure()
    pylab.plot(x, y, '.b', label="Raw datapoints")
    pylab.xlabel('Year')
    pylab.ylabel('Degrees Celsius')
    titlestr = titlestr + "\ngraph of linear regressed polynomial to fit data model of\n"

    for i, model in enumerate(models):
        model_degree = len(model)-1
        predicted_y = numpy.array(pylab.polyval(model, x))
        model_r_squared = r_squared(y, predicted_y)
        pylab.plot(x, predicted_y,
                        label="Linear regressed polynomial to fit the data of degree " + str(model_degree))
        titlestr += "Degree " + str(model_degree) + ", which has an RMSE value of " + str(model_r_squared) + "\n"
        if model_degree <= 1:
            model_se_over_slope = se_over_slope(x, y, predicted_y, model)
            titlestr += "and a se-over-slope (if following value is below 0.5 its good) of\n" + str(model_se_over_slope)+"\n"

    pylab.title(titlestr)
    leg = pylab.legend(loc='upper center')
    pylab.show()

def testFits(models, degrees, xVals, yVals, title):
    pylab.plot(xVals, yVals, 'o', label='Data')
    for i in range(len(models)):
        estYVals = pylab.polyval(models[i], xVals)
        error = r_squared(yVals, estYVals)
        pylab.plot(xVals, estYVals,
                   label='Fit of degree ' \
                         + str(degrees[i]) \
                         + ', R2 = ' + str(round(error, 5)))
    pylab.legend(loc='best')
    pylab.title(title)
    pylab.show()


    #
    # for i, model in enumerate(models):
    #     model_degree = len(model)-1
    #     predicted_y = numpy.asarray(pylab.polyval(model, x))
    #     model_r_squared = r_squared(y, predicted_y)
    #
    #
    #     pylab.figure()
    #     pylab.plot(x, y, '.b', label="Raw datapoints")
    #     pylab.plot(x, predicted_y, '-r', label="Linear regressed polynomial to fit the data of degree " + str(model_degree))
    #     pylab.xlabel('Year')
    #     pylab.ylabel('Degrees Celsius')
    #     titlestr = titlestr + "\ngraph of linear regressed polynomial to fit data model of degree " + str(model_degree) + ". This degree of model has an R2 value of "+str(model_r_squared)
    #     if model_degree <= 1:
    #         model_se_over_slope = se_over_slope(x, y, predicted_y, model)
    #         titlestr += "\nand a se-over-slope (if following value is below 0.5 its good) " + str(model_se_over_slope)

    #     pylab.title(titlestr)
    #     leg = pylab.legend(loc='upper center')
    #     pylab.show()

        



def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    # TODO
    final = []
    for year in years:
        citytemps = pylab.array([climate.get_yearly_temp(city, year) for city in multi_cities])
        final.append(citytemps.mean())
    return pylab.array(final)



def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    # TODOW
    return [sum(vals) / len(vals) for vals in [(y[max(i - window_length + 1, 0): i + 1]) for i in range(len(y))]]

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    # TODO
    error = sum([(y[i]-estimated[i])**2 for i in range(len(y))]) / len(y)
    return error ** (1/2)

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # TODO
    final = []  # holds the SD for each year
    for year in years:
        city_temps = []
        leap_temps = []
        for city in multi_cities:
            city_daily_temps = climate.get_yearly_temp(city, year)  # get the daily temperatures for each year and city
            if len(city_daily_temps) == 366:  # handling leap days
                leap_temps.append(city_daily_temps[59])
                city_daily_temps = pylab.delete(city_daily_temps, 59)
            city_temps.append(city_daily_temps)

        np_city_temps = numpy.array(city_temps)  # 2d array of daily temps; a 365 list for each city
        # print(len(city_temps[0]))
        averaged_city_temps = [numpy.sum(np_city_temps, axis=0)/len(np_city_temps) for i in range(len(np_city_temps))] #  wow i had to figure out what axis was LOL
        averaged_city_temps = averaged_city_temps[0].tolist()
        if len(leap_temps) != 0:
            leap_average = sum(leap_temps) / len(leap_temps)
            averaged_city_temps.insert(59, leap_average)
        year_std = numpy.std(averaged_city_temps)
        final.append(year_std)
        # print(averaged_city_temps, "|", averaged_city_temps[59], len(averaged_city_temps))
    print(final, len(final))
    return final




def generate_line_textures():
    pass # ill do this tomorrow

def evaluate_models_on_testing(x, y, models, titlestr=""):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # TODO

    pylab.figure()
    pylab.plot(x, y, '.b', label="Raw datapoints")
    pylab.xlabel('Year')
    pylab.ylabel('Degrees Celsius')
    titlestr = titlestr + "\ngraph of linear regressed polynomial to fit data model of degree "

    for i, model in enumerate(models):
        model_degree = len(model)-1
        predicted_y = numpy.array(pylab.polyval(model, x))
        model_rmse = rmse(y, predicted_y)
        pylab.plot(x, predicted_y,
                        label="Linear regressed polynomial to fit the data of degree " + str(model_degree))
        titlestr += str(model_degree) + ", which has an RMSE value of " + str(model_rmse) + "\n"

    pylab.title(titlestr)
    leg = pylab.legend(loc='upper center')
    pylab.show()

    # for i, model in enumerate(models):
    #     model_degree = len(model) - 1
    #     predicted_y = numpy.array(pylab.polyval(model, x))
    #     model_rmse = rmse(y, predicted_y)
    #
    #
    #     pylab.figure()
    #     pylab.plot(x, y, '.b', label="Raw datapoints")
    #     pylab.plot(x, predicted_y, '-r',
    #                label="Linear regressed polynomial to fit the data of degree " + str(model_degree))
    #     pylab.xlabel('Year')
    #     pylab.ylabel('Degrees Celsius')
    #     titlestr = titlestr + "\ngraph of linear regressed polynomial to fit data model of degree " + str(
    #         model_degree) + ". This degree of model has an RMSE value of " + str(model_rmse)
    #
    #     if model_degree <= 1:
    #         model_se_over_slope = se_over_slope(x, y, predicted_y, model)
    #         titlestr += "\nand a se-over-slope (if following value is below 0.5 its good) " + str(model_se_over_slope)
    #
    #     pylab.title(titlestr)
    #     leg = pylab.legend(loc='upper center')
    #     pylab.show()

if __name__ == '__main__':

    c = Climate("data.csv")

    # --------------------------------------------------- #
    # UNCOMMWENT EACH PART TO GET THE PLOT FOR EACH PART. #
    # --------------------------------------------------- #

    # Part A.4

    # 4.I: daily temperature
    # samplesx = []
    # samplesy = []
    # for year in TRAINING_INTERVAL:
    #     samplesy.append(c.get_daily_temp("NEW YORK", 1, 10, year))
    #     samplesx.append(year)
    #
    # x, y = pylab.array(samplesx), pylab.array(samplesy)
    # model = generate_models(x, y, [1])
    # evaluate_models_on_training(x, y, model, "NYC Jan 10 Temperatures")

    # 4.II: annual temperature
    # samplesx = []
    # samplesy = []
    # for year in TRAINING_INTERVAL:
    #     year_temps = c.get_yearly_temp("NEW YORK", year)
    #     samplesy.append(year_temps.mean())
    #     samplesx.append(year)
    #
    # x, y = pylab.array(samplesx), pylab.array(samplesy)
    # model = generate_models(x, y, [1])
    # evaluate_models_on_training(x, y, model, "NYC Average Annual Temperatures")

    # Part B
    # samplesy = gen_cities_avg(c, CITIES, TRAINING_INTERVAL)
    # samplesx = [year for year in TRAINING_INTERVAL]
    # x, y = pylab.array(samplesx), pylab.array(samplesy)
    #
    # model = generate_models(samplesx, samplesy, [1])
    # evaluate_models_on_training(x, y, model, "National Average Annual Temperature")

    # Part C
    # TODO: replace this line with your code

    # samplesx = [year for year in TRAINING_INTERVAL]
    # yeartemps = gen_cities_avg(c, CITIES, TRAINING_INTERVAL)
    # samplesy = moving_average(yeartemps, 5)
    #
    # x, y = pylab.array(samplesx), pylab.array(samplesy)
    #
    # model = generate_models(x, y, [1])
    # evaluate_models_on_training(x, y, model, "5 year moving average of US data")

    # Part D.2
    # training_samples_x = [year for year in TRAINING_INTERVAL]
    # year_temps = gen_cities_avg(c, CITIES, TRAINING_INTERVAL)
    # training_samples_y = moving_average(year_temps, 5)
    # training_x, training_y = pylab.array(training_samples_x), pylab.array(training_samples_y)
    #
    # models = generate_models(training_x, training_y, [1, 2, 20])
    # evaluate_models_on_training(training_x, training_y, models, "Model fitted to training data of USA moving average ")
    #
    # testing_samples_x = [year for year in TESTING_INTERVAL]
    # testing_year_temps = gen_cities_avg(c, CITIES, TESTING_INTERVAL)
    # testing_samples_y = moving_average(testing_year_temps, 5)
    # testing_x, testing_y = pylab.array(testing_samples_x), pylab.array(testing_samples_y)
    #
    # evaluate_models_on_testing(testing_x, testing_y, models, "Testing the models from the training interval on the texting interval.")

    # Part E
    std_devs = gen_std_devs(c, CITIES, TRAINING_INTERVAL)
    training_samples_y = moving_average(std_devs, 5)
    training_samples_x = [year for year in TRAINING_INTERVAL]
    training_x, training_y = pylab.array(training_samples_x), pylab.array(training_samples_y)

    models = generate_models(training_x, training_y, [1])
    evaluate_models_on_training(training_x, training_y, models, "Stardard deviations of USA annual temperature data")

# DONE! this pset was super super long, but I am finally finished!
