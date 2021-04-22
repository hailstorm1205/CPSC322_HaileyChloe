import math
import csv

# Splits dataset into test and train data
def train_test_split(X, y, test_size):
    num_instances = len(X) 
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size 

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Calculates distance between two points
def calc_distance(x1, x2, test1, test2):
    return math.sqrt(((x1 - test1) ** 2) + ((x2 - test2) ** 2))

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "N/A" or row[col_index].count > 0:
            col.append(row[col_index])
    return col

# Converts strings to numerical values
def convert_to_numeric(values):
    for i in range(len(values)):
        for j in range(len(values[i])):
            numeric_value = float(values[i][j])
            values[i][j] = numeric_value
    return values

# Reads a table from a CSV file
def read_table(filename):
    infile = open(filename, "r")
    the_reader = csv.reader(infile, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    infile.close()
    header = table[0]
    del table[0]
    return table, header

# Pretty prints a 2D array
def pretty_print(array):
    for val in array:
        for item in val:
            print(item, end=" ")
        print()