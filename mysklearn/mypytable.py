'''
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: #2
Date Last Updated: 2/18/21
Bonus?: No, I have not attempted the bonus.

Description: This file creates the class MyPyTable that uses a list of data with a header and 
    manipulates that data in various ways.
'''

import copy
import csv
import mysklearn.myutils as myutils
import numpy as np
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        #Determine # rows
        rows = len(self.data)
        #Determine # columns
        if(rows!=0):
            cols = len(self.data[0])
        else:
            cols = 0

        return rows, cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        #Validates variable "col_identifier" type
        if(type(col_identifier) is str or type(col_identifier) is int):
            #print("valid")
            pass
        else:
            raise ValueError("Value needs to be string or integer")

        #Validates variable "col_identifier" validity
        index = -1
        if(type(col_identifier) is str):
            #string --> determine if column name is valid
            for x in range(len(self.column_names)):
                if(self.column_names[x] == col_identifier):
                    index = x
                    break
            #-1 index means the column name wasn't found
            if(index == -1):
                raise ValueError("Column name not found")
        else:
            #int --> determine if index is valid
            if(self.get_shape()[1] > col_identifier):
                index = col_identifier
            else:
                raise ValueError("Index not found")

        #If indicated, rows with missing values ("NA") will be removed
        if(include_missing_values == False):
            self.remove_rows_with_missing_values()

        #Creation of new table with just one column
        new_table = []
        for x in self.data:
            for z in range(len(x)):
                if(z==index):
                    new_table.append(x[z])

        return new_table

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        #Search through all data
        for x in self.data:
            for i in range(len(x)):
                try:
                    float(x[i]) #check if can be type float
                    x[i] = float(x[i]) #convert to type float
                except:
                    #Value cannot be converted to float
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        #Add elements not being removed to new table
        new_table = []
        found = False
        for search in self.data:
            for row in rows_to_drop:
                if(row == search):
                    found = True
            if(found == False):
                new_table.append(search)
            else:
                found = False

        #deep copy new_table to self.data
        self.data.clear()
        self.data = copy.deepcopy(new_table)

    def drop_col(self, col_identifier):

        #Validates variable "col_identifier" type
        if(type(col_identifier) is str or type(col_identifier) is int):
            #print("valid")
            pass
        else:
            raise ValueError("Value needs to be string or integer")

        #Validates variable "col_identifier" validity
        if(type(col_identifier) is str):
            #string --> determine if column name is valid
            try:
                index = self.column_names.index(col_identifier)
            except:
                raise ValueError("Column name not found")
        else:
            #int --> determine if index is valid
            if(self.get_shape()[1] > col_identifier):
                index = col_identifier
            else:
                raise ValueError("Index not found")

        #Creation of new table with given column removed
        new_table = copy.deepcopy(self.data)
        for i,row in enumerate(new_table):
            row.pop(index)

        return new_table


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        self.data.clear()
        with open(filename) as csvFile:
            reader = csv.reader(csvFile)
            self.column_names = next(reader)
            for row in reader:
                self.data.append(row)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """

        #Initialize Variables
        duplicates_found = []
        saved_keys = []
        prev = 0

        #Create a list that allows for easy searching of key attributes
        for row in self.data:
            col = []
            for key in key_column_names:
                col.append(row[self.column_names.index(key)])
            saved_keys.append(col)

        #Find Duplicates
        for row in range(len(saved_keys)):
            while(prev != row):
                if(saved_keys[prev] == saved_keys[row]):
                    duplicates_found.append(self.data[row])
                    break #stops searching for duplicate value
                prev = prev+1
            prev = 0

        return duplicates_found

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        #Remove rows with "NA" in 1+ columns
        new_table = []
        value_found = False
        for x in self.data:
            for i in range(len(x)):
                if(x[i] == "NA" or x[i] == ""):
                    value_found = True
            if(value_found==False):
                new_table.append(x)
            else:
                value_found = False

        #Remove all missing rows from table
        self.data = new_table
        
    def remove_rows_with_missing_values_from_given_cols(self, col_names):
        """Remove rows from the table data that contain a missing value ("NA").
        
        Args:
            col_names(list of str): names of columns to be checked for missing values
        """

        #Remove rows with "NA" in 1+ columns
        new_table = []
        value_found = False
        for row in self.data:
            for col in col_names:
                index = self.column_names.index(col)
                if(row[index] == "" or row[index] == "NA"):
                    value_found = True
                    break
            if(value_found==False):
                new_table.append(row)
            else:
                value_found = False

        #Remove all missing rows from table
        self.data = new_table

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        #Check to make sure column values can be averaged??

        #Calculate column average
        col_sum = 0
        col_total = 0
        for row in self.data:
            for i in range(len(row)):
                if(self.column_names[i] == col_name and row[i] != "NA"):
                    col_sum = col_sum + row[i]
                    col_total = col_total + 1

        col_avg = round(col_sum / col_total, 1) #round value to 2 decimals

        #Find all values that are "NA"
        #Replace "NA" with column average
        for row in self.data:
            for i in range(len(row)):
                if(self.column_names[i] == col_name and row[i] == "NA"):
                    row[i] = col_avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed [attribute, min, max, mid, avg, median].
        """

        newList = []
        #Compute Summary Statistics
        for col in col_names:
            colList = self.get_column(col,False)
            if(len(colList)!=0): #only calculate if not empty
                #Calculate Median
                medVal = 0
                colList.sort()
                if(len(colList)%2 == 0): #even
                    medVal = (colList[int(len(colList)/2)]+colList[int((len(colList)-1)/2)])/2
                else: #odd
                    medVal = colList[int((len(colList)-1)/2)]
                #Calculate other summary stats & Add to new list
                newList.append([col,min(colList),max(colList),(min(colList)+max(colList))/2,np.round(sum(colList)/len(colList),3),medVal])

        print(tabulate(newList,["attr","min","max","mid","avg","med"]))

        return MyPyTable(["attr","min","max","mid","avg","med"],newList)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        table1 = [] #key columns in self
        table2 = [] #key columns in other_table
        #Get necessary key columns for easier searching
        for element in self.data:
            newList = []
            for key in key_column_names:
                newList.append(element[self.column_names.index(key)])
            table1.append(newList)
        for element in other_table.data:
            newList = []
            for key in key_column_names:
                newList.append(element[other_table.column_names.index(key)])
            table2.append(newList)

        #Create new header with values from self and other_table
        newHeader = copy.deepcopy(self.column_names)
        for val in other_table.column_names:
            try:
                newHeader.index(val)
            except:
                newHeader.extend([val])

        newTable = []
        #Iterate through first table
        for i in range(len(table1)):
            row = []
            #Iterate through second table
            for j in range(len(table2)):
                #If key values are equal...
                if(table1[i]==table2[j]):
                    #Add data from self to row
                    row.extend(self.data[i])
                    #Add data from other_table to row
                    #Only if not already in row list
                    for x in range(len(other_table.column_names)):
                        col = other_table.column_names[x]
                        try:
                            #Index is part of table1
                            index = self.column_names.index(col)
                        except:
                            #Index is NOT part of table1
                            #Needs to be added
                            index = other_table.column_names.index(col)
                            row.append(other_table.data[j][index])
                    break
            #Only add new row if not empty
            if(len(row) != 0):
                newTable.append(row)

        return MyPyTable(newHeader,newTable)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        table1 = [] # [[key1,key2,...,kn],...,[...]]
        table2 = []
        #Get necessary rows for columns
        for element in self.data:
            newList = []
            for key in key_column_names:
                newList.append(element[self.column_names.index(key)])
            table1.append(newList)
        for element in other_table.data:
            newList = []
            for key in key_column_names:
                newList.append(element[other_table.column_names.index(key)])
            table2.append(newList)

        #Create new header
        newHeader = copy.deepcopy(self.column_names)
        for val in other_table.column_names:
            try:
                newHeader.index(val)
            except:
                newHeader.extend([val])

        newTable = []
        #Iterate through first table
        flag = False
        for i in range(len(table1)):
            row = []
            #Iterate through second table
            for j in range(len(table2)):
                #If key values are equal...
                row = []
                if(table1[i]==table2[j]):
                    flag = True
                    #Add data from self to row
                    row.extend(self.data[i])
                    #Add data from other_table to row
                    #Only if not already in row list
                    for x in range(len(other_table.column_names)):
                        col = other_table.column_names[x]
                        try:
                            #Index is part of table1
                            index = self.column_names.index(col)
                        except:
                            #Index is NOT part of table1
                            #Needs to be added
                            index = other_table.column_names.index(col)
                            row.append(other_table.data[j][index])
                    #Only add new row if not empty
                    if(len(row) != 0):
                        newTable.append(row)
                    else:
                        row = copy.deepcopy(self.data[i])
                        while(len(row)<len(newHeader)):
                            row.append("NA")
                        newTable.append(row)
            #Add missed rows from self to newTable
            if(flag == False):
                row = copy.deepcopy(self.data[i])
                while(len(row)<len(newHeader)):
                    row.append("NA")
                newTable.append(row)
            else:
                flag = False

        new_header_indexes = []
        #Find indexes of cols in other_table in newTable
        for i in range(len(other_table.column_names)):
            col = other_table.column_names[i]
            new_header_indexes.append(newHeader.index(col))

        #For every value in other_table
        flag = False
        colFound = False
        for i in range(len(other_table.data)):
            #Check to see if the row is in newTable
            row = other_table.data[i]
            row_toBe_added = []
            for j in range(len(newTable)):
                newList = []
                for x in new_header_indexes:
                    newList.append(newTable[j][x])
                if(row == newList):
                    flag = True
                    break
            if(flag == False):
                #Row wasn't added
                #Needs to be added
                for headerCol in newHeader:
                    for otherCol in range(len(other_table.column_names)):
                        if(headerCol == other_table.column_names[otherCol]):
                            #Add data
                            row_toBe_added.append(other_table.data[i][otherCol])
                            colFound = True
                            break
                    if(colFound == False):
                        row_toBe_added.append("NA")
                    else:
                        colFound = False
                newTable.append(row_toBe_added)
            else:
                flag = False

        return MyPyTable(newHeader,newTable)