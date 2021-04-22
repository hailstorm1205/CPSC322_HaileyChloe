'''
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: #3
Date Last Updated: February 25th, 2021
Bonus?: I attempted the first bonus. Couldn't figure out how to properly plot the data, though.
Sources Used:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    
Desceiption: This file plots the data for video games, automobiles, and movies.
'''

import matplotlib.pyplot as plt
import os
import mysklearn.myutils as utils
import importlib
import numpy as np

def bar_chart(name,column):
    """Creates a bar chart.

        Args:
            name (str): value being sorted into categories
            column (list): list of data used to get information from

        Notes:
            Used in VGSales.ipynb
    """
    
    #Define variables
    x_vals = []
    x = []
    y = []
    
    #Put values in lists
    for value in column:
        try:
            index = x_vals.index(value)
            #Value is in x_vals list
            y[index] = y[index] + 1
        except:
            #Value not in x_vals list yet
            x_vals.append(value)
            x.append(len(x))
            y.append(1)
            
    #Use matplotlib to create graph
    plt.figure(figsize = (10,5))
    plt.bar(x,y)
    
    #Create titles
    plt.title("{} Frequency Diagram".format(name))
    plt.xlabel("Categories")
    plt.ylabel("Frequency")
    
    #Customize the x-tick labels
    plt.xticks(x, x_vals, rotation=45, horizontalalignment="right")
    
    #Show the plot
    plt.show()

def pie_chart(table):
    """Creates a pie chart.

        Args:
            table(MyPyTable): object from mypytable.py module that holds the necessary data.

        Notes:
            Used in VGSales.ipynb
    """
    
    #Define Variables
    columnList = []
    values = ["NA", "EU", "JP", "Other", "Global"]
    sums = []
    #Get the column of every percentage included (+ total)
    #Also initialize sums list to 0.0 for every row
    for value in values:
        columnList.append(table.get_column("{}_Sales".format(value),False))
        sums.append(0.0)

    #Calculate sums
    for row in range(len(columnList)):
        for i in range(len(columnList[row])):
            sums[row] = round((sums[row] + columnList[row][i]),2)

    #Find percentages and set x/y lists
    x = values[0:len(values)-1]
    y = []
    for i in range(len(sums)-1):
        y.append(sums[i]/sums[len(sums)-1])

    #Create pie chart
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%", normalize=False)
    plt.title("Percent of Global Sales")
    plt.show()
    
def easy_bar_chart(column):
    """Creates bar chart. First approach to AutoData Step 2.

        Args:
            column(list): list of data used to get information from

        Notes:
            Used in AutoData.ipynb
    """
    
    #Use matplotlib to create graph
    x,y = utils.approach_one(column)
    plt.figure()
    plt.bar(x,y)
    
    #Create labels
    plt.title("Ratings Frequency Diagram")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    
    #Customize the x-tick labels
    plt.xticks(x, x, rotation=45, horizontalalignment="right")
    
    #Show the plot
    plt.show()
    
def approach_two(column):
    """Creates bar chart. Second approach to AutoData Step 2.

        Args:
            column(list): list of data used to get information from

        Notes:
            Used in AutoData.ipynb
    """
    
    #Computer cutoffs and frequencies
    cutoffs = utils.compute_equal_width_cutoffs(column,5)
    freqs = utils.compute_bin_frequencies(column,cutoffs)
    
    #Use matplotlib to create graph
    plt.figure()
    plt.bar(cutoffs[:-1],freqs,width=cutoffs[1]-cutoffs[0],align="edge");
    
    #Create labels
    plt.title("MPG Frequency Diagram")
    plt.xlabel("MPG Values")
    plt.ylabel("Frequency")
    
    #Show the plot
    plt.show()

def createHistogram(name, column):
    """Creates histogram.

        Args:
            name(str): value being plotted.
            column(list): list of data used to get information from

        Notes:
            Used in AutoData.ipynb
    """
    #Remove percentages
    column = utils.removePercentage(column)
    
    #Use matplotlib to create histogram
    plt.figure()
    plt.hist(column, bins=10)
    
    #Create title
    plt.title("{} Histogram".format(name))
    plt.xlabel("{} Ratings".format(name))
    plt.ylabel("Frequency")
    
    #Show diagram
    plt.show()
    
def createScatterPlot(name,x,y):
    """Creates scatter plot

        Args:
            name(str): name of value being plotted against MPG
            x(list): list of values used for the x-axis
            y(list): list of values used for the y-axis

        Notes:
            Used in AutoData.ipynb
    """
    
    #Calculate slope
    m, b = utils.compute_slope_intercept(x,y)
    
    #Use matplotlib to create histogram
    plt.figure()
    plt.scatter(x,y)
    plt.plot([min(x), max(x)], [m*min(x)+b, m*max(x)+b], c="r",lw=5)
    
    #Create title
    plt.title("{} vs. MPG Scatter Plot".format(name))
    plt.xlabel(name)
    plt.ylabel("MPG")
    
    #Annotate
    #Calculate correlation coefficient
    corrCoeff = utils.calculateCorrCoeff(x,y)
    plt.annotate("$corCoef = {}$".format(str(corrCoeff)),xy=(.85,.9),xycoords="axes fraction", horizontalalignment="center",color="blue")
    #Calculate covariance
    covariance = utils.calculateCovariance(x,y)
    plt.annotate("$cov = {}$".format(str(covariance)),xy=(.85,.8),xycoords="axes fraction", horizontalalignment="center",color="blue")
    
    #Show Diagram
    plt.show()
    
def movieBarChart(table):
    """Creates bar chart for movies data.

        Args:
            table(MyPyTable): object from mypytable.py module that holds the necessary data.

        Notes:
            Used in Movies.ipynb
    """
    
    #Determine how many movies are on each service
    x,y = utils.movieFrequencies(table)
    
    #Use matplotlib to create bar chart
    fig, ax = plt.subplots() #necessary to annotate heights properly
    rects = ax.bar(x,y)
    
    #Create labels
    ax.set_title("Movies per Streaming Service\nFrequency Diagram")
    ax.set_xlabel("Streaming Services")
    ax.set_ylabel("Frequency")
    
    #Annotate
    #Add heights to each bar to show the exact number of movies
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),xy=(rect.get_x()+rect.get_width()/2,height), xytext=(0,3),textcoords="offset points",ha="center",va="top")
        
    fig.tight_layout()
    
    #Show Diagram
    plt.show()
    
def moviePieChart(table):
    """Creates pie chart for movies data.

        Args:
            table(MyPyTable): object from mypytable.py module that holds the necessary data.

        Notes:
            Used in Movies.ipynb
    """
    
    #Determine how many movies are on each service
    x,y = utils.movieFrequencies(table)
    
    #Create pie chart
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title("Percentage of Movies\nOn Each Streaming Service")
    plt.show()
    
def movieScatterPlot(x_title,x,y_title,y):
    """Creates scatter plot for movies data.

        Args:
            x_title(str): name of values on x-axis
            x(list): list of x-axis values
            y_title(str): name of values on y-axis
            y(list): list of y-axis values

        Notes:
            Used in Movies.ipynb
    """
    
    #Remove percentages
    x = utils.removePercentage(x)
    y = utils.removePercentage(y)
    
    #Calculate slope
    m, b = utils.compute_slope_intercept(x,y)
    
    #Use matplotlib to create histogram
    plt.figure() #figsize=(10,10))
    plt.scatter(x,y)
    plt.plot([min(x), max(x)], [m*min(x)+b, m*max(x)+b], c="r",lw=5)
    
    #Create labels
    plt.title("{} vs. {} Scatter Plot".format(x_title,y_title))
    plt.xlabel("{} Ratings".format(x_title))
    plt.ylabel("{} Ratings".format(y_title))
    
    #Show Diagram
    plt.show()
    
def movieBoxPlot(ratingName,ratingColumn,genreColumn):
    """Creates box plot for movies data.

        Args:
            ratingName(str): either "IMDb" or "Rotten Tomatoes" (ie. rating being plotted)
            ratingColumn(list): list of movie ratings
            genreColumn(list): list of genres for each movie

        Notes:
            Used in Movies.ipynb
    """
    #Removes percent sign and multiplies by 0.01
    #only if a percent is found
    ratingColumn = utils.removePercentage(ratingColumn)
    
    #Define Variables
    x = []
    y = []
    
    #Assign values to lists
    for i in range(len(genreColumn)):
        genreList = genreColumn[i].split(",")
        for genre in genreList:
            try:
                index = x.index(genre)
                y[index].append(ratingColumn[i])
            except:
                x.append(genre)
                y.append([ratingColumn[i]])
                
    #Use matplotlib to create boxplot
    plt.figure(figsize=(10,5))
    plt.boxplot(y)
    
    #Create labels
    plt.xticks(list(range(1,len(x)+1)),x,rotation=45)
    plt.title("{} Ratings by Genre".format(ratingName))
    plt.xlabel("Genres")
    plt.ylabel("{} Ratings".format(ratingName))
    
    #Show Diagram
    plt.show()
    
def grouped_bar_chart(genreIndex, table):
    pass
    '''#Define variables
    regions = ["NA", "EU", "JP", "Other", "Global"]
    x = []
    y = []
    for row in table.data:
        for i in range(len(regions)):
            salesIndex = table.column_names.index("{}_Sales".format(regions[i]))
            try:
                xIndex = x.index(row[genreIndex])
                if(i>(len(y[xIndex])-1)):
                    y[xIndex].append([row[salesIndex]])
                else:
                    y[xIndex][i].append(row[salesIndex])
            except:
                x.append(row[genreIndex])
                y.append([[row[salesIndex]]])
                
    x2 = np.arange(len(x))
    print(x2)
    width = 0.15
    
    fig, ax = plt.subplots()
    rects = [0]*len(regions)
    means = [0]*len(regions)
    
    j=0
    for category in y:
        for i in range(len(category)):
            val = round(sum(category[i])/len(category[i]),2)
            if(means[i]==0):
                means[i]=[val]
            else:
                means[i].append(val)
                
    print(means)
    for j in range(len(means)):           
        for i in range(len(regions)):
            rects[i] = ax.bar(x2[j]-width/len(regions),means[i],width,label=regions[i])
    
    #Create labels
    ax.set_title("Sales From Each Region\nBy Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Frequency")
    #ax.set_xticklabels(x)
    ax.legend()
    
    #Annotate
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),xy=(rect.get_x()+rect.get_width()/2,height), xytext=(0,3),textcoords="offset points",ha="center",va="top")
        
    fig.tight_layout()
    
    #Show Diagram
    plt.show()'''
            
    