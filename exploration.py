import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read in data from csv file
heart_dat = pd.read_csv("data.csv", encoding='cp1252')

#Names for each col in data set
age = heart_dat['ï»¿age']
sex = heart_dat['sex'] 
cp = heart_dat['cp']
chol = heart_dat['chol']
bp = heart_dat['trestbps']

"""Method that calculates the summary values of each column in the data frame"""
def summary_values(col):
    print("Mean" , col.mean())
    print("Median" , col.median())
    print("Mode" , col.mode())
    print("Max" , col.max())
    print("Min" , col.min())
    print("Range" , (col.max())-(col.min()))

"""Barplot based on inputs of column name for x-axis, data frame, and the plot number to be printed"""
def one_barplot(x_val, dat, num):
    sns.countplot(x=x_val, data=dat)
    if num == 1:
        plt.savefig('Plot1.pdf')
    else:
        plt.savefig('Plot2.pdf')
    plt.show()

"""Barplot based on inputs of column name for x-axis, value that the data will be shown in terms of, data frame, and the plot number to be printed"""
def two_barplot(x_val, hues, dat, num):
    sns.countplot(x=x_val, hue=hues, data=dat)
    if num == 1:
        plt.savefig('Plot3.pdf')
    else:
        plt.savefig('Plot4.pdf')
    plt.show()

"""Scatterplot based on inputs of data frame, column name for x-axis, and the column name for the y-axis"""
def one_scatplot(dat, x_val, y_val):
    sns.scatterplot(data=dat, x=x_val, y=y_val)
    plt.savefig('Plot5.pdf')
    plt.show()

"""Scatterplot based on inputs of data frame, column name for x-axis, and the value that the data will be shown in terms of"""
def two_scatplot(dat, x_val, y_val, hues):
    sns.scatterplot(data=dat, x=x_val, y=y_val, hue=hues)
    plt.savefig('Plot6.pdf')
    plt.show()

def main():
    one_barplot(age, heart_dat, 1)
    one_barplot(sex, heart_dat, 2)

    two_barplot(age, sex, heart_dat, 1)
    two_barplot(age, cp, heart_dat, 2)

    one_scatplot(heart_dat, age, chol)
    
    two_scatplot(heart_dat, age, chol, sex)

    #summary_values(bp)

###################################

if __name__ == '__main__':
    main()
