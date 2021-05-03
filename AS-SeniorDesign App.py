"""
#########################################################
## Anton Skurdal & Sairam Sunkam                       ##
## University of North Dakota - CSCI 493 Senior Design ##
## Spring 2021                                         ##
## Real Time Machine Learning for SCADA Systems        ##
## Final GUI Program                                   ##
#########################################################
"""



"""
#############
## IMPORTS ##
#############
"""
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import seaborn as sns
import sys
import time
import math
import csv
import os
import calendar
from datetime import datetime as dt


import tkinter as tk
from tkinter import*
from tkinter import IntVar
from tkinter import StringVar
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
#import tensorflow as tf

import threading
import subprocess
#from subprocess import Popen, PIPE

#Suppress Warning
pd.options.mode.chained_assignment = None


"""
#############
# FUNCTIONS #
#############
"""
#help(tk.Tk)
def injectAttacks(df, dayList, ax, ay, outputType, weight, forceAttack):

	"""
	============================================================================
	DESCRIPTION:
	Takes a dateframe to inject attacks into, a list of days to inject, the
	average fitted attack values and output types for each time of the year,
	the amount to weight the injected attacks by, and whether or not to force
	attacks when the normal value is greater. Then, uses these inputs to loop
	through the day list, isolate a single day and modify it's values using
	the weighted average, and inject the attack data into a copy of the original
	data frame. The function returns this injected data


	Takes a dataframe and creates a list of NORMAL energy consumption and time 
	values for a single day based on the String date, a list of INJECTED ATTACK
	average time & energy consumption values for the entire year as well as
	their output values, and a weight. Uses all this data to create a list of
	weighted attacks for each day by averaging the time's specific attack value
	for the year with the (weighted) day's time value for the year.
	============================================================================
	INPUTS:
	df - dataframe into which attacks will be injected, in minutes form
	dayList - date of the day to inject as a String or list of Strings
	ax - average time values for each hour for the attacks
	ay - average energy consumption values calculated using the attack fit
	outputType - tells whether a row in ax/ay is an Attack or Normal
	weight - how many times the normal column values are included in the average
	forceAttack - boolean to control whether attack values replace normal values
	              that are greater than the weighted average (default = False)
	============================================================================
	OUTPUTS:
	- dataframe with data injected
	============================================================================
	"""

	#Create String list of days
	days = []
	days.extend(dayList)

	#Create dataframe to modify
	modData = df

	for i in days:
		#print(i)
		day = i
		#Get day's date
		data = df.groupby(['Date'])
		date = data.get_group(day)





		#Convert it into minutes - convertTime needs work
		#date = convertTime(date, 2)
		#print("Converting Time Column to type: Integer Tag")
		df['Time'] = np.arange(len(df['Time']))		
		
		date['Time'] = date['Time']*10
		#Reset the index
		date = date.reset_index(drop = True)



		#Print info
		#print("\nOriginal Day Data")
		#print(date)
		#print(len(date['Time']))

		#Get attack values and insert their weighted average into dataframe
		for i in range(len(date)):
			if (outputType[i] == "Attack"):
				#print("[Avg Att]: {} [Avg Nml]: {}".format(ay[i], date['Energy_consumption'][i]))

				#Compute weighted average
				weightedAvg = ((date['Energy_consumption'][i] * weight) + ay[i])/(1 + weight)

				#print("WeightedAvg: ", weightedAvg)

				#Check if lower weighted averages are replacing normal data
				if (forceAttack == False and date['Energy_consumption'][i] > weightedAvg):

					#Print info
					"""
					print("\nAttack not forced:")
					print("Index: ", i)
					print("[Avg Nml]: ", date['Energy_consumption'][i])
					print("WeightedAvg: ", weightedAvg)
					print(date.iloc[[i]])
					"""
					continue

				#Print info
				"""
				print("\nAttack Injected:")
				print("Index: ", i)
				print("[Avg Nml]: ", date['Energy_consumption'][i])
				print("WeightedAvg: ", weightedAvg)
				"""

				#Inject attack
				date['Energy_consumption'][i] = weightedAvg
				date['Output'][i] = outputType[i]

				"""
				print(date.iloc[[i]])
				"""

		#print("\n\nBEFORE:")
		#print(modData.loc[0:5])
		#print(modData.loc[26260:26265])

		modData[modData['Date'] == day] = modData[modData['Date'] == day].assign(Output = date['Output'].values)
		modData[modData['Date'] == day] = modData[modData['Date'] == day].assign(Energy_consumption = date['Energy_consumption'].values)

		#print("\n\nAFTER:")
		#print(modData.loc[0:5])
		#print(modData.loc[26260:26265])		

	#modData= modData.reset_index(drop = True)
	return modData

def chooseDays(df, percent, specific):

	"""
	Parameter Guide:

	df - dataframe containing days

	percent - percentage of days to pick
	**Percent must be an integer between 0 and 100***

	specific - string or list of strings containing specific dates
	"""	

	#Get unique dates
	dates = df.Date.unique()

	#Declare list to hold chosen days
	days = []

	#Append single specific date to days
	if (type(specific) == str):
		for i in dates:
			if (i == specific):
				days.append(i)
		#print(days)

	#Append list of specific dates to days
	elif (type(specific) == list):
		for i in specific:
			for j in dates:
				if (j == i):
					days.append(j)
		#print(days)		

	#Choose random dates and append them to days
	else:
		temp = np.random.uniform(0, 1, len(dates)) <= (percent/100)
		days = dates[temp == True]
		#print(days)

	return days

def getInjectAttacks(start, stop, percent, c, x, y):

	"""
	============================================================================
	DESCRIPTION:
	Calculates an equation based on yearly average energy consumption 
	values that will be used (along with a weighted average) to inject
	attacks into a certain day or list of days; all within a specified
	range and by a specified percent.
	============================================================================
	INPUTS:
	start - start time in minutes (inclusive) (int between 0 and 1430)
	stop - stop time in minutes (inclusive) (int between 0 and 1430)
	percent - percent to multiply data points by (int between 0 and 100)
	c - list of coefficients
	x - list of time values (specific day)
	y - list of energy consumption values (specific day)
	============================================================================
	OUTPUTS:
	xatt - list of all times
	yatt - list of energy consumption values for each value of xatt
	dataType - list containing data about each row of xatt and yatt
	============================================================================
	"""

	#Format percent increase
	p = 1 + percent/100
	#print(p)

	#Reverse the coefficents so that the function works properly
	c = c[::-1]

	#Create a function out of the coefficents
	f = np.poly1d(c)

	xtest = 1200
	ytest = f(xtest)
	#plt.scatter(xtest, ytest)

	xatt = x
	#print(x_data)
	yatt = []
	dataType = []
	for i in range(len(xatt)):

		if (xatt[i] >= start and xatt[i] <= stop):
			yatt.append(f(xatt[i])*p)
			dataType.append("Attack")
		else:
			yatt.append(f(xatt[i]))
			dataType.append("Normal")


	#print(xatt, yatt)
	return xatt, yatt, dataType

def printEquation(coefs):
	fn = "f(x) ="
	power = len(coefs) - 1
	for i in coefs:
		if (power == len(coefs) - 1):
			c = "({:.2e})".format(i)
			fn += "{"+  c  +"x^"+str(power)+"}"
		else:
			c = "({:.2e})".format(i)
			fn += " + {"+c+"x^"+str(power)+"}"
		power -= 1
	#print(fn)
	return(fn)

def getHourlyAvg(df):

	"""
	============================================================================
	DESCRIPTION:
	Takes a dataframe and groups it by Time, as well as average energy
	consumption and attack values for that time across the entire dataset.
	Converts them into DateTime objects and sorts them into the proper order.
	============================================================================
	INPUTS:
	df - dataframe from which to extract and reformat data
	============================================================================
	OUTPUTS:
	hourAvg - the sorted year average column values for each unique time
	============================================================================
	"""	
	
	
	#print(df.head(5))
	#Group the data by time and get an average value for each time
	hourAvg = df.groupby(['Time'], as_index=False).mean()

	#Convert the times into datetime objects using strptime
	for i in hourAvg.index:
		temp = dt.strptime(hourAvg.at[i, 'Time'], '%I:%M:%S %p').time()
		hourAvg.at[i, 'Time'] = temp	

	#Sort the hourAvg by time
	hourAvg = hourAvg.sort_values(by='Time', ascending = True, ignore_index=True)

	#Return the hourAvg dataframe
	return hourAvg

def curveFit(df, x, y):
	"""
	============================================================================
	DESCRIPTION:
	Takes dataframe or two lists and curve fits the points inside.
	============================================================================
	INPUTS:
	df - dataframe with columns to curve fit
	x - list of x values
	y - list of y values
	============================================================================
	OUTPUTS:
	coefs - curve fit coefficients (highest power first)
	x - list of values
	y - list of values
	============================================================================
	"""
	
	if (x != None and y != None):
		x = x
		y = y
	else:
		#Define axis
		x = list(df['Time'])
		y = list(df['Energy_consumption'])




	#Plot the data
	#plt.scatter(x, y, label = 'Energy_consumption')
	#plt.scatter(x, y2, label = 'Cost')	

	"""
	#################
	# CURVE FITTING #
	#################
	"""
	#Create a line of evenly spaced numbers over the interval len(x)
	x_fitLine = np.linspace(x[0], x[-1], num = len(x) * 10)

	#Fit the line and save the coefficients
	coefs = poly.polyfit(x, y, 9)

	#Use the values of x_fitLine as inputs for the polynomial
	#function with given coefficients to create y_fitLine
	y_fitLine = poly.polyval(x_fitLine, coefs)

	#Plot the fitLine
	#plt.plot(x_fitLine, y_fitLine)
	
	#printEquation(coefs)
	
	#plt.show()
	

	return coefs, x, y

def AttackInject(fname, master, seed, start, stop, percentAttackIncrease, percentDays, weight, forceAttack, eqFrame):
	
	#Grid configuration
	for col in range(1):
		master.grid_columnconfigure(col, weight=1)
	for row in range(1):
		master.grid_rowconfigure(row, minsize=50)	
	master.grid_rowconfigure(1, weight=0)

	#Console Label
	for child in master.winfo_children():
		child.destroy()		
	var = StringVar()
	console = CustomLabel(master, text="", textvariable=var, font=['Arial', 12, 'bold'], justify = "left", anchor="nw", background="#212121", wraplength=500)
	console.grid(row = 0, column = 0, rowspan = 2, columnspan = 2, sticky = "NSEW", padx = 10, pady = 10)	

	#Message
	var.set(var.get() + "Loading " + fname)
	
	
	#CONTROLS
	start = int(start)
	stop = int(stop)
	percentAttackIncrease = int(percentAttackIncrease)
	percentDays = int(percentDays)
	weight = int(weight)
	forceAttack = bool(forceAttack)
	
	#Declare a seed for the RNG
	seed = int(seed)	
	
	"""
	#CONTROLS
	start = 400
	stop = 1400
	percentAttackIncrease = 50
	percentDays = 50
	weight = 3
	forceAttack = True
	
	#Declare a seed for the RNG
	seed = 2
	"""
	np.random.seed(seed)	
	
	#Read in dataset
	dataset = pd.read_csv(fname)
	
	var.set(var.get() + "\nRemoving attacks... ")
	#Remove attacks
	data = dataset[dataset.Output == "Normal"]
	modData = data
	
	var.set(var.get() + "\nCalculating hourly average values... ")
	#Get hourly average values
	hourlyAvg = getHourlyAvg(data)
	
	#Convert the hourly average from DateTime into minutes
	minutesList = []
	for i in hourlyAvg['Time']:
		temp = str(i).split(':')
		mins = 0
		mins += int(temp[0])*60 + int(temp[1])
		minutesList.append(mins)
	hourlyAvg['Time'] = minutesList	
	
	var.set(var.get() + "\nCurve fitting normal data... ")
	#Fit the normal average and save the coefficients and columns as lists
	normalCoefs, nx, ny = curveFit(hourlyAvg, None, None)
	#printEquation(normalCoefs)
	
	var.set(var.get() + "\nRetrieving attacks... " + fname)
	#Inject attacks by percentage into list using function
	#nx = times (mins), ny = energy consumption
	ax, ay, attackDataType = getInjectAttacks(start, stop, percentAttackIncrease, normalCoefs, nx, ny)
	
	var.set(var.get() + "\nCurve fitting attack data...")
	#Fit the attack data
	attackCoefs, ax, ay = curveFit(None, ax, ay)	
	
	var.set(var.get() + "\nChoosing injection days...")
	#Choose which days in which to inject attacks
	attackDays = chooseDays(data, percentDays, None)
	
	#################################################################
	var.set(var.get() + "\nPrinting equations...")
	#Print Equations
	nEq = printEquation(normalCoefs)
	aEq = printEquation(attackCoefs)
	
	
	
	#Console Label
	eq = StringVar()
	console2 = CustomLabel(eqFrame, text="", textvariable = eq, font = ['Arial', 11, 'bold'], justify = "left", anchor="w", background = "#424242", wraplength = 800)
	console2.grid(sticky = "NSEW", padx=4, pady=4)		
	eq.set(eq.get() + "Normal Fit Equation:\n" + nEq + "\nAttack Fit Equation:\n" + aEq)
	console2.update()
	
	
	
	
	
	var.set(var.get() + "\nCreating plot...")
	#Create plot
	x = np.linspace(minutesList[0], minutesList[-1], num=len(minutesList)*10)
	#print("\n" + str(list(nx)))
	nfit = poly.polyval(x, normalCoefs)
	afit = poly.polyval(x, attackCoefs)
	#plt.plot(x, nfit, label = "Normal Curve Fit")
	#plt.plot(x, afit, label = "Attack Curve Fit")
	#plt.legend()
	##Time(minutes) - Energy Consumption
	#plt.show()
	
	title = "Curve Fit of Normal and Attack Averages"
	xlabel = "Time (minutes)"
	ylabel = "Energy Consumption"
	
	#Figure creation
	figure = Figure()
	plot = figure.add_subplot(111)
	
	#Plot Formatting
	plot.set_facecolor('#E0E0E0')
	figure.patch.set_color('#424242')
	plot.spines['bottom'].set_color('#BDBDBD')
	plot.spines['top'].set_color('#BDBDBD')
	plot.spines['right'].set_color('#BDBDBD')
	plot.spines['left'].set_color('#BDBDBD')
	plot.set_title(title, fontsize=14, color='#BDBDBD')
	plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
	plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD', labelpad = 15)
	plot.tick_params(axis='x', colors='#BDBDBD')
	plot.tick_params(axis='y', colors='#BDBDBD')
	mpl.rcParams['font.sans-serif'] = "Arial"
	mpl.rcParams['font.family'] = "sans-serif"	
	
	#Plot graph
	plot.plot(x, nfit, label = "Normal Curve Fit")
	plot.plot(x, afit, label = "Attack Curve Fit")
	plot.legend()			

	#Place graph
	canvas = FigureCanvasTkAgg(figure, master)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
	toolbarFrame = Frame(master=master)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	toolbar.config(background = "#424242")
	toolbar._message_label.config(background='#424242')
	
	for button in toolbar.winfo_children():
		button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
	toolbar.update()			
	canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")		
	
	
	
	#################################################################
	
	#Inject attacks into day(s)
	modData = data
	modData = injectAttacks(modData, attackDays, ax, ay, attackDataType, weight, forceAttack)	
	
	#Create CSV file with unique name
	vlen = len(modData['Output'])
	
	normCt = np.sum(modData['Output'] == "Normal")
	attCt = np.sum(modData['Output'] == "Attack")
	normPct = normCt/vlen * 100
	attPct = attCt/vlen * 100
	
	print("Normal: {0:2.2f}%".format(normPct))
	print("Attack: {0:2.2f}%".format(attPct))
	attPct = int(attPct)
	
	
	csvName = "FDIinject"
	csvDateTime = dt.now()
	csvPredicate = csvDateTime.strftime("%Y-%m-%d %H.%M.%S")
	csvName += " " + str(attPct) + " " + csvPredicate+ ".csv"
	print("[File Name]: ", csvName)
	modData.to_csv(csvName, index = False)

def RunRandomForest(fname, master, seedVar, trainVar, n_jobsVar, startVar, stopVar, numVar, max_featuresVar, critVar, max_depthVar, min_ssplitVar, min_sleafVar, bootstrapVar):
	
	#Grid configuration
	for col in range(1):
		master.grid_columnconfigure(col, weight=1)
	for row in range(1):
		master.grid_rowconfigure(row, minsize=50)	
	master.grid_rowconfigure(1, weight=0)
	
	#Console Label
	for child in master.winfo_children():
		child.destroy()	
	var = StringVar()
	console = CustomLabel(master, text="", textvariable = var, font = ['Arial', 12, 'bold'], justify = "left", anchor="nw", background = "#212121", wraplength = 600)
	console.grid(row = 0, column = 0, rowspan = 2, columnspan = 2, sticky = "NSEW", padx = (10, 10), pady = (10, 10))	
	
	#Message
	var.set(var.get() + "Setting parameters...")
	console.update()
	
	#Formatting
	np.set_printoptions(threshold=sys.maxsize)
	pd.set_option('display.max_rows', 100000)
	
	##############
	# PARAMETERS #
	##############
	#Create dataset object
	#fname = "FDIinject 49P.csv"
	print(fname.get())
	dataset = pd.read_csv(fname.get())	
	
	#Seed - seed for random number generator
	seed = int(seedVar.get())
	np.random.seed(seed)	
	
	#Training Percent - percentage of data to select for training
	trainVar = int(trainVar.get()) / 100
	
	#Number of Jobs - Number of processor cores to use
	n_jobsVar = int(n_jobsVar.get())
	
	#Number of Estimators - number of trees in forest - #10 was best out of [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000]
	startVar = int(startVar.get())
	stopVar = int(stopVar.get())
	numVar = int(numVar.get())
	n_estimators = [int(x) for x in np.linspace(start = startVar,stop = stopVar, num = numVar)] 
	
	#Max Features - number of features to consider at every split - #max_features = ['auto'] # auto > sqrt and log2
	temp = str(max_featuresVar.get())
	temp = temp.replace(" ", "")
	max_featuresVar = []
	max_featuresVar.append(temp)
	max_features = max_featuresVar
	
	#Criterion - measures the quality of the split (gini or entropy) - #criterion= ['entropy','gini'] # 'entropy' better than 'gini index'
	temp = str(critVar.get())
	temp = temp.replace(" ", "")
	temp = temp.split(",")
	critVar = temp
	criterion = critVar
	
	#Max Depth - maximum number of levels in tree - #max_depth = [10] #10 best out of [2,5,10,15,20,50]
	temp = int(max_depthVar.get())
	max_depthVar = []
	max_depthVar.append(temp)
	max_depth = max_depthVar
	
	#Min Samples Split - minimum number of samples required to split a node - #min_samples_split = [10] #10 best out of [2,5,10,20]	
	temp = int(min_ssplitVar.get())
	min_ssplitVar = []
	min_ssplitVar.append(temp)
	min_samples_split = min_ssplitVar
	
	#Min Samples Leaf - minimum number of samples required at each leaf node - #min_samples_leaf = [15] #15 best out of [5,10,15,20,25,30]
	temp = int(min_sleafVar.get())
	min_sleafVar = []
	min_sleafVar.append(temp)
	min_samples_leaf = min_sleafVar
	
	#Bootstrap - method of selecting samples for training each tree - #bootstrap = [True] #False was best out [True, False]
	temp = bool(bootstrapVar.get())
	bootstrapVar = []
	bootstrapVar.append(temp)
	bootstrap = bootstrapVar
	
	#Print info and pause if true
	printPause = False
	if (printPause == True):
		print("Seed: ", seed)
		print("Training Percent: ", trainVar)
		print("Number of Jobs: ", n_jobsVar)
		print("Number of Estimators:")
		print("\tstart: ", startVar)
		print("\tstop: ", stopVar)
		print("\tnum: ", numVar)
		print("Max Features: ", max_featuresVar)
		print("Criterion: ", critVar)
		print("Max Depth: ", max_depthVar)
		print("Minimum Samples Split: ", min_ssplitVar)
		print("Minimum Samples Leaf: ", min_sleafVar)
		print("Bootstrap: ", bootstrapVar)
		os.system("PAUSE")
	
	#Message
	var.set(var.get() + "\nImporting dataset...")
	console.update()	
	
	
	
	#Message
	var.set(var.get() + "\nFormatting data...")
	console.update()	
	
	#Drop bad columns
	del dataset["Date"]
	del dataset["Time"]
	
	#Create Test and Train data
	dataset['is_train'] = np.random.uniform(0, 1, len(dataset)) <= trainVar
	pd.set_option('display.max_columns', None)
	
	#Create train and test datasets
	train = dataset[dataset['is_train'] == True]
	test = dataset[dataset['is_train'] == False]
	
	#Show lengths of each
	print("Training length: ", len(train))
	print("Testing length: ", len(test))
	
	#Create list of feature column's names
	features = dataset.columns[:2]
	#print("Features",features)
	
	#Converting each output possibility into digits
	y_train = pd.factorize(train['Output'])[0]
	
	#Creating random forest classifier
	var.set(var.get() + "\nCreating Random Forest Classifier...")
	console.update()	
	clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
	
	#Training the classifier
	var.set(var.get() + "\nTraining the Random Forest Classifier...")
	console.update()	
	clf.fit(train[features], y_train)
	
	#Get predictions
	var.set(var.get() + "\nFetching predictions...")
	console.update()	
	preds = clf.predict(test[features])
	
	#Map output integers to names
	preds_read = []
	for x in range(len(preds)):
		#print(preds[x])
		if preds[x] == 0:
			preds_read.append("Attack")
		else:
			preds_read.append("Normal")
	
	#Calculate prediction accuracy
	var.set(var.get() + "\nCalculating prediction accuracy...")
	console.update()	
	accuracy = []
	correct = 0
	incorrect = 0
	test2 = test['Output'].to_numpy()
	for x in range(len(preds_read)):
		if preds_read[x] == test2[x]:
			correct = correct + 1
		else:
			incorrect = incorrect + 1
	test_actual_labels = test['Output'].to_numpy()
	test_actual = []
	for label in test_actual_labels:
		if (label == 'Attack'):
			test_actual.append(0)
		else:
			test_actual.append(1)
	y_test = np.array(test_actual)
	
	
	#Print accuracy numbers
	total = correct + incorrect
	percentRight = correct/total
	percentAccuracy = "Accuracy:% 0.3f%%" %(percentRight * 100)
	var.set(var.get() + "\nPREDICTIONS:")
	var.set(var.get() + "\n\tCorrect: " + str(correct))
	var.set(var.get() + "\n\tIncorrect: " + str(incorrect))
	var.set(var.get() + "\n\tAccuracy: " + str(percentAccuracy))
	console.update()	
	
	#Create the Param Grid
	var.set(var.get() + "\nCreating Parameter Grid...")
	console.update()	
	param_grid = {'n_estimators' : n_estimators,
		      'criterion' : criterion,
		      'max_features' : max_features,
		     'max_depth' : max_depth,
		     'min_samples_split' : min_samples_split,
		     'min_samples_leaf': min_samples_leaf,
		     'bootstrap' : bootstrap}
	var.set(var.get() + "\nParameter Grid:")
	var.set(var.get() + "\n\t" + str(param_grid))	
	console.update()	
	
	#Run GridSearchCV
	var.set(var.get() + "\nRunning GridSearchCV...")
	console.update()	
	clf_Grid = GridSearchCV(clf,param_grid = param_grid, cv=3, verbose=10, n_jobs = n_jobsVar)
	clf_Grid.fit(train[features], y_train)
	clf_Grid.best_params_
	
	#print(f'Train Accuracy - : {clf_Grid.score(train[features], y_train)}')
	#print(f'Test Accuracy - : {clf_Grid.score(test[features], y_test)}')
	testAccuracy = f'Test Accuracy - : {clf_Grid.score(test[features], y_test)}'
	var.set(var.get() + "\nTest Accuracy:")
	var.set(var.get() + "\n\t" + testAccuracy)
	console.update()		
	
	#Create GridSearchCV Results Table
	var.set(var.get() + "\nCreating GridSearchCV Results Table...")
	console.update()
	table = pd.pivot_table(pd.DataFrame(clf_Grid.cv_results_),
			       values='mean_test_score', index='param_n_estimators', 
	    columns='param_criterion')
	var.set(var.get() + "\nGridSearchCV Results Table:")
	var.set(var.get() + "\n\t" + str(table))
	console.update()	
	
	#Create the heatmap and apply it to the GUI
	var.set(var.get() + "\nCreating heatmap...")
	console.update()
	#time.sleep(2)
	
	#Destroy console to make room for figure
	console.destroy()
	
	c = sns.palplot(sns.color_palette("dark:#339900"))
	print(c)
	
	#Create fiugre containing heatmap
	figure = Figure()
	ax = figure.subplots()
	plot = sns.heatmap(table, ax = ax, cbar_kws={'label': 'Accuracy %'}, cmap="Greens_r")
	#plot = sns.heatmap(table, ax = ax, cbar_kws={'label': 'Accuracy %'}, cmap=sns.palplot(sns.color_palette("dark:#339900")))
	
	
	title = "Criterion Accuracy for Number of Estimators"
	xlabel = "Criterion"
	ylabel = "Number of Estimators"
	
	
	#Plot Formatting
	plot.set_facecolor('#E0E0E0')
	figure.patch.set_color('#424242')
	
	plot.spines['bottom'].set_color('#BDBDBD')
	plot.spines['top'].set_color('#BDBDBD')
	plot.spines['right'].set_color('#BDBDBD')
	plot.spines['left'].set_color('#BDBDBD')
	plot.set_title(title, fontsize=16, color='#BDBDBD')
	plot.set_xlabel(xlabel, fontsize=14, color='#BDBDBD')	
	plot.set_ylabel(ylabel, fontsize=14, color='#BDBDBD')
	
	plot.figure.axes[-1].yaxis.label.set_size(14)
	plot.figure.axes[-1].yaxis.label.set_color("#E0E0E0")
	plot.figure.axes[-1].tick_params(labelcolor = "#E0E0E0", color = "#E0E0E0")
	#for tick_label in plot.figure.axes[-1].yaxis.get_yticklabels():
		#tick_label.set_color("white")
		#tick_label.set_fontsize("30")
	
	plot.tick_params(colors='#BDBDBD')
	#plot.tick_params(axis='y', colors='#BDBDBD')
	mpl.rcParams['font.sans-serif'] = "Arial"
	mpl.rcParams['font.family'] = "sans-serif"	
	"""
	#Plot graph
	plot.scatter(x, y, label = 'Energy_consumption')
	plot.scatter(x, y2, label = 'Cost')
	plot.legend()			
	"""
	#Place graph
	canvas = FigureCanvasTkAgg(figure, master)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
	toolbarFrame = Frame(master=master)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	toolbar.config(background = "#424242")
	toolbar._message_label.config(background='#424242')
	
	for button in toolbar.winfo_children():
		button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
	toolbar.update()			
	canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")	
	
	"""
	#Create canvas for figure
	canvas = FigureCanvasTkAgg(figure, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, padx=(0,0), pady = (0, 0), sticky = "NSEW")
	
	#Recreate Matplotlib toolbar
	toolbarFrame = Frame(master=master)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	toolbar.update()
	
	#Place toolbar on window
	canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")	
	"""
	
def get_attributes(widget):
	widg = widget
	keys = widg.keys()
	for key in keys:
		print("Attribute: {:<20}".format(key), end=' ')
		value = widg[key]
		vtype = type(value)
		print('Type: {:<30} Value: {}'.format(str(vtype), value))	

"""
####################
# END OF FUNCTIONS #
####################
"""



"""
###################
# ROOT APP WINDOW #
###################
"""
class App(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		
		self.title_font = tkfont.Font(family = 'Arial', size = 14, weight = "bold")
		self.title('AS-SS Real-Time Machine Learning for SCADA Application')
		self.geometry("1820x720")
		self.resizable(0, 0)
		self['background'] = '#222222'
		self['bd'] = 10
		self.iconphoto(True, tk.PhotoImage(file='icon.png'))		
		
		
		
		"""COMBO BOX"""
		style = ttk.Style()
		style.theme_use('clam')
		"""
		style.configure('TCombobox', background = '#424242', foreground = '#E0E0E0',
				padding = 5, borderwidth=0, bordercolor = '#424242', highlightbackground = '#616161', highlightcolor = '#616161', highlightthickness = 1,
				relief=FLAT, state='readonly', arrowcolor='#E0E0E0',font = ('Arial', 20, 'bold'),
			lightcolor='#424242', darkcolor='#424242')
		
		style.map('TCombobox', background=[('active','#616161'), ('focus', '#616161')], lightcolor=[('focus','#616161')],
			  darkcolor=[('focus','#616161')], bordercolor=[('focus', '#616161')],
			  selectbackground=[('readonly', 'none')], selectforeground=[('readonly', '#E0E0E0')],fieldbackground=[('focus','#616161'),('readonly', '#424242'), ('active', 'red')])
		"""
		style.configure('TCombobox', arrowcolor = '#E0E0E0')
		style.configure('TCombobox', arrowsize = 20)
		
		style.map('TCombobox', foreground=[('','#E0E0E0')])
		style.map('TCombobox', fieldbackground=[('hover', '#616161'),('','#424242')])
		style.map('TCombobox', bordercolor=[('hover', '#616161'),('','#424242')])
		style.map('TCombobox', selectbackground=[('','none')])
		style.map('TCombobox', selectforeground=[('','#E0E0E0')])
		
		style.map('TCombobox', background=[('hover', '#616161'),('','#424242')])
		style.map('TCombobox', highlightcolor=[('hover', '#616161'),('','#424242')])
		
		style.map('TCombobox', darkcolor=[('hover', '#616161'),('','#424242')])
		style.map('TCombobox', lightcolor=[('hover', '#616161'),('','#424242')])		
		
		style.configure('TLabelframe', bordercolor = "red", borderwidth = 10)
		#style.map('TLabelframe', borderwidth = [('', 10)])
		style.configure('TLabelframe.Label', borderwidth = 10)
		style.configure('TLabelframe', backgroundcolor = [('', "blue")])
		
		
		#style.configure('TLabelframe.Label', font = ('Arial', 16, 'normal'))
		#style.map('TLabelframe.Label', foreground=[('','#E0E0E0')])
		#style.map('TLabelframe.Label', background = [('', self['background'])])
		
		#style.theme_use('clam') #only theme to handle bordercolor for labelframe
	
		style.configure('info1.TLabelframe', bordercolor='#009699',background=self['background'], lightcolor = "#009699", darkcolor = "#009699")
		style.configure('info1.TLabelframe.Label', foreground='#E0E0E0',background=self['background'], font = ('Arial', 16))
		
		style.configure('orange.TLabelframe', bordercolor='#bf4300',background=self['background'], lightcolor = "#bf4300", darkcolor = "#bf4300")
		style.configure('orange.TLabelframe.Label', foreground='#E0E0E0',background=self['background'], font = ('Arial', 16))
		
		style.configure('green.TLabelframe', bordercolor='#339900',background=self['background'], lightcolor = "#339900", darkcolor = "#339900")
		style.configure('green.TLabelframe.Label', foreground='#E0E0E0',background=self['background'], font = ('Arial', 16))		
		
		#style.map('TCombobox', background=[('','#FF0000')])
		#style.map('TCombobox', background=[('disabled','#FF0000')])
		
		
		#style.map('TCombobox', fieldbackground=[('hover','#616161')])
		#style.map('TCombobox', selectbackground=[('hover', 'green')])
		#style.map('TCombobox', selectforeground=[('hover', 'blue')])
		
		fontStyle = ('Arial', 12, 'normal')
		
		
		
		self.option_add('*TCombobox.font', fontStyle)
		self.option_add('*TCombobox*Listbox.font', fontStyle)
		self.option_add('*TCombobox*Listbox.background', '#424242')
		self.option_add('*TCombobox*Listbox.foreground', '#E0E0E0')
		self.option_add('*TCombobox*Listbox.selectBackground', '#616161')
		self.option_add('*TCombobox*Listbox.selectForeground', '#E0E0E0')
		
		
		
		
		
		#Main menu frame
		menu_frame = tk.Frame(self, bg='#222222', width = 200, height = 700)
		menu_frame.grid(row = 0, column = 0, sticky = "NSEW")
		menu_frame.grid_propagate(False)
		
		
		menu_frame.grid_rowconfigure(0, minsize=100)
		menu_frame.grid_columnconfigure(0, minsize=200)
		
		menu_frame.grid_rowconfigure(1, minsize=200)
		menu_frame.grid_columnconfigure(1, minsize=200)
		
		menu_frame.grid_rowconfigure(2, minsize=200)
		menu_frame.grid_columnconfigure(2, minsize=200)
		
		menu_frame.grid_rowconfigure(3, minsize=200)
		menu_frame.grid_columnconfigure(3, minsize=200)			
		
		#Menu Buttons
		splash_button = CustomButton(menu_frame, text="Information", justify="center", bd=0, font = ('Arial', 18), activebackground="#616161", command=lambda: self.show_frame("Splash"))
		tab1_button = CustomButton(menu_frame, text="Dataset\nAnalyzer", justify="center", bd=0, font = ('Arial', 18),activebackground="#009699", command=lambda: self.show_frame("TabOne"))
		tab2_button = CustomButton(menu_frame, text="Attack\nInjector", justify="center", bd=0, font = ('Arial', 18),activebackground="#BF4300", command=lambda: self.show_frame("TabTwo"))
		tab3_button = CustomButton(menu_frame, text="Random\nForest\nClassifier", justify="center", bd=0, font = ('Arial', 18),activebackground="#339900", command=lambda: self.show_frame("TabThree"))
		
		
		
		
		#splash_button.grid(row = 0, column = 0, sticky="NSEW", padx=(0,2), pady=(0,0))	
		#tab1_button.grid(row = 1, column = 0, sticky="NSEW", padx=(0,2), pady=(2,0))
		#tab2_button.grid(row = 2, column = 0, sticky="NSEW", padx=(0,2), pady=(2,2))
		#tab3_button.grid(row = 3, column = 0, sticky="NSEW", padx=(0,2), pady=(0,0))
		splash_button.grid(row = 0, column = 0, sticky="NSEW", padx=(0,10), pady=(0,10))	
		tab1_button.grid(row = 1, column = 0, sticky="NSEW", padx=(0,10), pady=(0,10))
		tab2_button.grid(row = 2, column = 0, sticky="NSEW", padx=(0,10), pady=(0,10))
		tab3_button.grid(row = 3, column = 0, sticky="NSEW", padx=(0,10), pady=(0,0))		
		
		
		#Main container for frame stacking
		container = tk.Frame(self, bg='green', width = 1600, height = 700)
		container.grid(row = 0, column = 1, sticky = "NSEW")
		container.grid_propagate(False)
		
		self.frames = {}
		for Tab in (Splash, TabOne, TabTwo, TabThree):
			page_name = Tab.__name__
			frame = Tab(parent = container, controller = self)
			self.frames[page_name] = frame
			
			#Put all of the pages in the same place
			#The frame on top will be visible
			frame.grid(row = 0, column = 0, sticky = "NSEW")
		
		self.show_frame("Splash")
	
	def show_frame(self, page_name):
		#Show the frames for the given page name
		frame = self.frames[page_name]
		frame.tkraise()

class CustomButton(tk.Button):
	def __init__(self, master, **kw):
		tk.Button.__init__(self, master=master, **kw)
		
		#self.defaultBackground = self["background"]
		#self.defaultBackground = "#424242"
		#self['font'] = 'Arial', 12, 'normal'
		
		if (self['font'] == "TkDefaultFont"):
			self['font'] = ['Arial', 12, 'normal']		
		else:
			self['font'] = self['font']
		
		self['background'] = "#424242"
		self['foreground'] = "#E0E0E0"
		self['activeforeground'] = "#E0E0E0"
		
		#self['highlight']
		
		self['borderwidth'] = 0
		
		self['relief'] = "solid"
		self['padx'] = 2
		self['pady'] = 2
		
		
		self.bind("<Enter>", self.on_enter)
		self.bind("<Leave>", self.on_leave)
		self.bind("<Button-1>", self.on_click)

	def on_enter(self, e):
		self['background'] = self['activebackground']
	
	def on_leave(self, e):
		#self['background'] = self.defaultBackground
		self['background'] = "#424242"
	
	def on_click(self, e):
		self['background'] = "#4E4E4E"
		self['relief'] = "sunken"

class CustomLabel(tk.Label):
	def __init__(self, master, **kw):
		tk.Label.__init__(self, master=master, **kw)
		#print(self['font'])
		if (self['background'] == "SystemButtonFace"):
			self['background'] = "#424242"
		else:
			self['background'] = self['background']
		
		if (self['font'] == "TkDefaultFont"):
			self['font'] = ['Arial', 12, 'normal']
		else:
			self['font'] = self['font']
			
		
		if (self['fg'] == "SystemButtonText"):
			self['fg'] = "#E0E0E0"
		else:
			self['fg'] = self['fg']		

class CustomCheckbutton(tk.Checkbutton):
	def __init__(self, master, **kw):
		tk.Checkbutton.__init__(self, master=master, **kw)
		self['background'] = "#424242"
		self['foreground'] = "#E0E0E0"
		self['selectcolor'] = "#009699"
		self['font'] = ['Arial', 12, 'normal']
		self['activebackground'] = "#616161"
		self['activeforeground'] = "#E0E0E0"

class CustomLabelFrame(tk.LabelFrame):
	def __init__(self, master, **kw):
		tk.LabelFrame.__init__(self, master=master, **kw)
		#print(self['font'])
		if (self['background'] == "SystemButtonFace"):
			self['background'] = master['background']
		else:
			self['background'] = self['background']
		
		if (self['font'] == "TkDefaultFont"):
			self['font'] = ['Arial', 16, 'normal']
		else:
			self['font'] = self['font']	
		
		#self['systemButtonFrame'] = "red"
		self['labelanchor'] = "n"
		self['fg'] = "#E0E0E0"
		self['relief'] = SOLID
		self['bd'] = 10
		self['highlightcolor'] = "#FFFFFF"
		self['highlightbackground'] = "#FFFFFF"
		self['highlightthickness'] = 10



"""
##########
# SPLASH #
##########
"""
class Splash(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg="#212121", width = 1600, height = 700)
		
		self.controller = controller
		self.grid_rowconfigure(0, minsize = 50)
		self.grid_rowconfigure(1, minsize = 650)
		for col in range(3):
			self.grid_columnconfigure(col, weight = 1)		
		self.grid_anchor('center')
		
		#Row 0
		title1 = CustomLabel(self, text = "Dataset Analyzer", background = "#212121", anchor = 'sw', font = ['Arial', 18, 'normal'], padx = 10)
		title1.grid(row = 0, column = 0, sticky = "NSEW", padx = (0, 5))
	
		title2 = CustomLabel(self, text = "Attack Injector", background = "#212121", anchor = 'sw', font = ['Arial', 18, 'normal'], padx = 10)
		title2.grid(row = 0, column = 1, sticky = "NSEW", padx = (5, 5))		
	
		title3 = CustomLabel(self, text = "Random Forest Classifier", background = "#212121", anchor = 'sw', font = ['Arial', 18, 'normal'], padx = 10)
		title3.grid(row = 0, column = 2, sticky = "NSEW", padx = (5, 10))		
		
		
		#Frame formatting
		l_frame = tk.Frame(self, bg="#009699")
		m_frame = tk.Frame(self, bg="#BF4300")
		r_frame = tk.Frame(self, bg="#339900")
	
		l_frame.grid(row = 1, column = 0, sticky = "NSEW", padx=[0, 5])
		l_frame.grid_propagate(False)
		
		m_frame.grid(row = 1, column = 1, sticky = "NSEW", padx=[5, 5])
		m_frame.grid_propagate(False)		
		
		r_frame.grid(row = 1, column = 2, sticky = "NSEW", padx=[5, 10])
		r_frame.grid_propagate(False)					
		
		
		##############
		# LEFT FRAME #
		##############
		l_frame.grid_rowconfigure(0, weight = 1)		
		l_frame.grid_columnconfigure(0, weight = 1)
		
		info1 = CustomLabel(l_frame, font = ['Arial', 14, 'normal'], anchor = 'nw', background = "#212121", padx = 10, pady = 10, justify = 'left',
		                    text = "  Dynamically creates graphs based on a given dataset.\n\n"+
		                    "\u2022 The graphs are configurable, with options including:\n\t-duplicate pruning\n\t-column selection\n\t-month selection\n\t-day selection"+
		                    "\n\t-classifier prediction\n\t-hourly averages")
		info1.grid(sticky = "NSEW", padx=2, pady = 2)		
		
		
		################
		# MIDDLE FRAME #
		################		
		m_frame.grid_rowconfigure(0, weight = 1)		
		m_frame.grid_columnconfigure(0, weight = 1)
		
		info2 = CustomLabel(m_frame, font = ['Arial', 14, 'normal'], anchor = 'nw', background = "#212121", padx = 10, pady = 10, justify = 'left',
		                    text = "  Injects attacks into a dataset based on user-defined\n"+
		                    "  parameters.\n\n"+
		                    "\u2022 Parameters include:\n\t-start time\n\t-stop time"+
		                    "\n\t-attack percent increase\n\t-percent injection days"+
		                    "\n\t-weight\n\t-force attacks\n\t-seed\n\n"
		                    "\u2022 Displays the curve-fit equation for the"+
		                    " normal average\n  hourly data and the attack hourly average"+
		                    " data.\n\n"+
		                    "\u2022 Displays a graph of the curve fit equations.\n\n"+
		                    "\u2022 Saves the modified dataset to the program's current\n  directory as a CSV file.")
		info2.grid(sticky = "NSEW", padx=2, pady = 2)	
		
		
		###############
		# RIGHT FRAME #
		###############		
		r_frame.grid_rowconfigure(0, weight = 1)		
		r_frame.grid_columnconfigure(0, weight = 1)	
		
		info3 = CustomLabel(r_frame, font = ['Arial', 14, 'normal'], anchor = 'nw', background = "#212121", padx = 10, pady = 10, justify = 'left',
		                    text = "  Predicts the output values of a given dataset using a \n  Random Forest Classifier.\n\n"+
		                    "\u2022 Optimizes hyperparameters using GridSearchCV.\n\n"+
		                    "\u2022 The parameter grid is configurable, with options including:"+
		                    "\n\t-seed\n\t-training percent\n\t-number of jobs\n\t-number of estimators"+
				    "\n\t-max features\n\t-criterion\n\t-max depth\n\t-min samples split"+
		                    "\n\t-min samples leaf\n\t-bootstrap")
		info3.grid(sticky = "NSEW", padx=2, pady = 2)		


"""
###########
# TAB ONE #
###########
"""
class TabOne(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg="#212121", width = 1600, height = 700)
		self.controller = controller		
		
		#Frame formatting
		l_frame = tk.Frame(self, bg='#212121', width = 800, height = 700)
		r_frame = tk.Frame(self, bg='#009699', width = 800, height = 700)
		
		l_frame.grid(row = 0, column = 0, sticky = "NSEW", padx=[0, 5])
		l_frame.grid_propagate(False)
		
		r_frame.grid(row = 0, column = 1, sticky = "NSEW", padx=[5, 0])
		r_frame.grid_propagate(False)			
		
		
		##############################
		# LEFT FRAME GRID FORMATTING #
		##############################
		for col in range(4):
			l_frame.grid_columnconfigure(col, weight=1)
		
		for row in range(10):
			#l_frame.grid_rowconfigure(row, minsize=60)
			l_frame.grid_rowconfigure(row, weight=1, minsize = 50)
			
		
		#Functions
		def load_file():
			file = filedialog.askopenfilename(filetypes = [('CSV files', '.csv')], title = "Dataset Selection", initialdir = os.getcwd())
			fname.set(os.path.split(file)[1])

		def handle_focus_in(_):
			E1.select_range(0, END)
			E1.config(fg='#E0E0E0', selectbackground = "#009699", highlightbackground = '#616161', highlightcolor = '#009699', highlightthickness = 1, borderwidth = 5)		
	
		def handle_return(_):
			fname.set(E1.get())

		def showStats():
			fname = E1.get()
			data = pd.read_csv(fname)
			
			print(prune.get())
			if (prune.get() == 1):
				dupeNum = len(data)				
				data = data.drop_duplicates(subset=['Energy_consumption', 'Cost'])
				dupeNum -= len(data)
			else:
				dupeNum = None
			
			text = "First five rows of dataset:\n" + str(data.head(5))
			text += "\n\nValue frequency:\n"
			for i in data.columns:
				text += "\n" + str(i) + ":\n"
				text += str(data[i].value_counts()) + "\n" 
				
			
			title = "'" + fname + "' Statistics"
			
			if (dupeNum != None):
				text += "\nNumber of duplicates:" + str(dupeNum)
				title += " (pruned duplicates)"
			
			messagebox.showinfo(title, text)	
		
		def plotYear(master):
			
			for child in master.winfo_children():
				child.destroy()
			
			data = pd.read_csv(E1.get())
			average = data.groupby(['Date'], as_index=False).mean()
			
			
			for i in average.index:
				temp = dt.strptime(average.at[i, 'Date'], '%m/%d/%Y').date()
				#temp = dt.strptime(average.at[i, 'Date'], '%m/%d/%Y') maybe change this to include time
				#print(temp)
				average.at[i, 'Date'] = temp
			
			x = average['Date']
			y = average['Energy_consumption']
			y2 = average['Cost']
			
			title = "Energy Consumption & Cost per Day"
			xlabel = "Date"
			ylabel = "Data"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)	
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
			plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD')
			plot.tick_params(axis='x', colors='#BDBDBD', rotation = 45)
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.scatter(x, y, label = 'Energy_consumption')
			plot.scatter(x, y2, label = 'Cost')
			plot.legend()			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")

		def plotMonth(master, monthnum):
			
			for child in master.winfo_children():
				child.destroy()			
			
			data = pd.read_csv(E1.get())
			average = data.groupby(['Date'], as_index=False).mean()			
			
			for i in average.index:
				temp = dt.strptime(average.at[i, 'Date'], '%m/%d/%Y').date()
				#temp = dt.strptime(average.at[i, 'Date'], '%m/%d/%Y') maybe change this to include time
				#print(temp)
				average.at[i, 'Date'] = temp				
			
			#monthnum = 12
			
			
			month = average[pd.to_datetime(average['Date']).dt.month == monthnum]
			
			x = month['Date']
			y = month['Energy_consumption']
			y2 = month['Cost']
			
			
			
			
			title = "Energy Consumption & Cost in " + calendar.month_name[monthnum]
			xlabel = "Day"
			ylabel = "Data"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)	
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
			plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD')
			plot.tick_params(axis='x', colors='#BDBDBD', rotation = 45)
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.scatter(x, y, label = 'Energy_consumption')
			plot.scatter(x, y2, label = 'Cost')
			#plot.xticks(rotation=45, ha='right')
			#plot.tight_layout()			
			plot.legend()			
			
			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")
		
		def plotDay(master, monthnum, daynum):
			
			for child in master.winfo_children():
				child.destroy()			
			
			data = pd.read_csv(E1.get())
			average = data.groupby(['Date'], as_index=False).mean()
			
			hourly = getHourlyAvg(data)
			
			hourlist = hourly['Time'].astype(str)
			ticks = []
			for x in range(0, 143, 6):
				#print(x)
				ticks.append(hourlist[x])
			ticks.append(hourlist[143])
			
			"""
			END OF HOUR LIST STUFF
			"""
			monthnum = int(monthnum)
			daynum = int(daynum)
			daytag = str(monthnum) + "/" + str(daynum) + "/2010"
			day = data.groupby("Date")
			day = day.get_group(daytag)
			x = hourlist
			y = day['Energy_consumption']
			y2 = day['Cost']
			
			title = "Energy Consumption & Cost on " + calendar.month_name[monthnum] + " " + str(daynum)
			xlabel = "Time"
			ylabel = "Data"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)	
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
			plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD')
			plot.tick_params(axis='x', colors='#BDBDBD', rotation = 45)
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.scatter(x, y, label = 'Energy_consumption')
			plot.scatter(x, y2, label = 'Cost')
			plot.set_xticks(ticks)
			plot.set_xticklabels(ticks, fontsize = 8)		
			plot.legend()			
			
			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")				

		def plotHour(master):
			
			for child in master.winfo_children():
				child.destroy()			
			
			data = pd.read_csv(E1.get())
			hourly = getHourlyAvg(data)
			
			
			hourlist = hourly['Time'].astype(str)
			ticks = []
			for x in range(0, 143, 6):
				#print(x)
				ticks.append(hourlist[x])
			ticks.append(hourlist[143])
			x = hourlist
			y = hourly['Energy_consumption']
			y2 = hourly['Cost']
			
			title = "Energy Consumption & Cost by Time"
			xlabel = "Time"
			ylabel = "Data"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
			plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD')
			plot.tick_params(axis='x', colors='#BDBDBD', rotation = 45)
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"
			
			#Plot graph
			plot.scatter(x, y, label = 'Energy_consumption')
			plot.scatter(x, y2, label = 'Cost')
			plot.set_xticks(ticks)
			plot.set_xticklabels(ticks, fontsize = 8)
			plot.legend()			

			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")				

		def rfClassify(master):
			
			for child in master.winfo_children():
				child.destroy()
			
			sel = rfChoice.get()
			
			if (sel == " FDI Training Set"):
				# Importing the dataset
				dataset = pd.read_csv(E1.get())

				#This basically reads in the index it is at and selects a location based off position
				#This will plot the positive or negative like a scatter plot
				#Reads the values and prepares them in an array for graphing
				#'loc' changes the values based on a condition
				dataset.loc[(dataset.Output == 'Attack'), 'Output'] = '1'
				dataset.loc[(dataset.Output == 'Normal'), 'Output'] = '0'
				X = dataset.iloc[:, [2, 3]].values
				y = dataset.iloc[:, 4].values
				
				# Splitting the dataset into the Training set and Test set
				#This takes a random sample of the dataset as the training set
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
			
			
				# Feature Scaling
				#This is so the scatter plot isn't massive, we want the dimensions to be set
				#We want all the plots to be the same size
				#Same size as necessary because the scales are custom set
				#We have to transform the data so the distribution has a mean value of 0 and a standard deviation of 1
				#e.g. Age has a standard devation of 1
				sc = StandardScaler()
				X_train = sc.fit_transform(X_train)
				X_test = sc.transform(X_test)

				# Fitting Random Forest Classification to the Training set
				# n-estimator controls the number of trees to built before the the prediction is made
				# More trees = better accuracy, but runs slower
				# This sets this up to be classified by allocating labeled memory for entropy trees
				classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
				classifier.fit(X_train, y_train)
			
				# Predicting the Test set results
				# This is a primitive way of predicting the output without using machine learning
				y_pred = classifier.predict(X_test)
			
				# Making the Confusion Matrix
				#This is a performance and error visualization --- aka accuracy of the subset
				cm = confusion_matrix(y_test, y_pred)		
				
				#Copies original array to preserve originals
				X_set, y_set = X_train, y_train
		
				##Controls the size of the mesh or graph bounds
				X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
						     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		
				#Figure containing plot
				figure = Figure()		
				plot = figure.add_subplot(111)
				
				#Plot contouring
				plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
						     alpha = 0.75, cmap = ListedColormap(('C0', 'C1')))
			
				#Sets the bounds for the data --- aka limits of the current axis
				plot.set_xlim(X1.min(), X1.max())
				plot.set_ylim(X2.min(), X2.max())		
		
				#Scatter plots each data point and color maps it
				for i, j in enumerate(np.unique(y_set)):
					print(i, j)
					plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('C0', 'C1'))(i), label = j)
					
				
				#Plot labels
				normal_handle = Line2D([0], [0], marker='o', color='none', label='Normal', markerfacecolor='C0', markeredgewidth=0, markersize=7)
				attack_handle = Line2D([0], [0], marker='o', color='none', label='Attack', markerfacecolor='C1', markeredgewidth=0, markersize=7)
				title = "Random Forest Classification (Training Set)"
				xlabel = "Energy Consumption"
				ylabel = "Cost"				
				
				#Plot Formatting
				plot.set_facecolor('#E0E0E0')
				figure.patch.set_color('#424242')
				plot.spines['bottom'].set_color('#BDBDBD')
				plot.spines['top'].set_color('#BDBDBD')
				plot.spines['right'].set_color('#BDBDBD')
				plot.spines['left'].set_color('#BDBDBD')
				plot.set_title(title, fontsize=14, color='#BDBDBD')
				plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
				plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD', labelpad = 15)
				plot.tick_params(axis='x', colors='#BDBDBD')
				plot.tick_params(axis='y', colors='#BDBDBD')
				mpl.rcParams['font.sans-serif'] = "Arial"
				mpl.rcParams['font.family'] = "sans-serif"	
				
				#Plot graph
				plot.legend(handles = [normal_handle, attack_handle])			
	
				#Place graph
				canvas = FigureCanvasTkAgg(figure, master)
				canvas.draw()
				canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
				toolbarFrame = Frame(master=master)
				toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
				toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
				toolbar.config(background = "#424242")
				toolbar._message_label.config(background='#424242')
				
				for button in toolbar.winfo_children():
					button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
				toolbar.update()			
				canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")
				
			elif (sel == " FDI Test Set"):

				# Importing the dataset
				dataset = pd.read_csv('FDIdataset.csv')
		
				#This basically reads in the index it is at and selects a location based off position
				#This will plot the positive or negative like a scatter plot
				#Reads the values and prepares them in an array for graphing
				#'loc' changes the values based on a condition
				dataset.loc[(dataset.Output == 'Attack'), 'Output'] = '1'
				dataset.loc[(dataset.Output == 'Normal'), 'Output'] = '0'
				X = dataset.iloc[:, [2, 3]].values
				y = dataset.iloc[:, 4].values
		
				# Splitting the dataset into the Training set and Test set
				#This takes a random sample of the dataset as the training set
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
		
				# Feature Scaling
				#This is so the scatter plot isn't massive, we want the dimensions to be set
				#We want all the plots to be the same size
				#Same size as necessary because the scales are custom set
				#We have to transform the data so the distribution has a mean value of 0 and a standard deviation of 1
				#e.g. Age has a standard devation of 1
				sc = StandardScaler()
				X_train = sc.fit_transform(X_train)
				X_test = sc.transform(X_test)
				
				# Fitting Random Forest Classification to the Training set
				# n-estimator controls the number of trees to built before the the prediction is made
				# More trees = better accuracy, but runs slower
				# This sets this up to be classified by allocating labeled memory for entropy trees
				classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
				classifier.fit(X_test, y_test)
		
				# Predicting the Test set results
				# This is a primitive way of predicting the output without using machine learning
				y_pred = classifier.predict(X_test)
		
				# Making the Confusion Matrix
				#This is a performance and error visualization --- aka accuracy of the subset
				cm = confusion_matrix(y_test, y_pred)		
		
				#Copies original array to preserve originals
				X_set, y_set = X_train, y_train
		
				##Controls the size of the mesh or graph bounds
				X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
						     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		
				#Figure containing plot
				figure = Figure()		
				plot = figure.add_subplot(111)
		
				#Plot contouring
				plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
					     alpha = 0.75, cmap = ListedColormap(('C0', 'C1')))
		
				#Sets the bounds for the data --- aka limits of the current axis
				plot.set_xlim(X1.min(), X1.max())
				plot.set_ylim(X2.min(), X2.max())		
				
				#Scatter plots each data point and color maps it
				for i, j in enumerate(np.unique(y_set)):
					print(i, j)
					plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('C0', 'C1'))(i), label = j)
		
				#Plot labels
				normal_handle = Line2D([0], [0], marker='o', color='none', label='Normal', markerfacecolor='C0', markeredgewidth=0, markersize=7)
				attack_handle = Line2D([0], [0], marker='o', color='none', label='Attack', markerfacecolor='C1', markeredgewidth=0, markersize=7)
				plot.legend(handles = [normal_handle, attack_handle])		
				title = "Random Forest Classification (Testing Set)"
				xlabel = "Energy Consumption"
				ylabel = "Cost"					
				
				#Plot Formatting
				plot.set_facecolor('#E0E0E0')
				figure.patch.set_color('#424242')
				plot.spines['bottom'].set_color('#BDBDBD')
				plot.spines['top'].set_color('#BDBDBD')
				plot.spines['right'].set_color('#BDBDBD')
				plot.spines['left'].set_color('#BDBDBD')
				plot.set_title(title, fontsize=14, color='#BDBDBD')
				plot.set_xlabel(xlabel, fontsize=12, color='#BDBDBD')	
				plot.set_ylabel(ylabel, fontsize=12, color='#BDBDBD', labelpad = 15)
				plot.tick_params(axis='x', colors='#BDBDBD')
				plot.tick_params(axis='y', colors='#BDBDBD')
				mpl.rcParams['font.sans-serif'] = "Arial"
				mpl.rcParams['font.family'] = "sans-serif"	
				
				#Plot graph
				plot.legend(handles = [normal_handle, attack_handle])			
	
				#Place graph
				canvas = FigureCanvasTkAgg(figure, master)
				canvas.draw()
				canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
				toolbarFrame = Frame(master=master)
				toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
				toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
				toolbar.config(background = "#424242")
				toolbar._message_label.config(background='#424242')
				
				for button in toolbar.winfo_children():
					button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
				toolbar.update()			
				canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")	
		
		def plotSelection(master):
			data = pd.read_csv(E1.get())
			pruned = False
			
			if (prune.get() == 1):
				dupeNum = len(data)
				data = data.drop_duplicates(subset=['Energy_consumption', 'Cost'])
				dupeNum -= len(data)
				print("Number of duplicates:", dupeNum)
				pruned = True
			
			sel = plotSel.get()
			if (sel == " Energy vs Cost"):
				plotEvC(master, data, pruned)
			elif (sel == " Energy Consumption"):
				plotE(master, data, pruned)
			elif (sel == " Cost"):
				plotC(master, data, pruned)
			elif (sel == " Energy to Cost Ratio"):
				plotRatio(master, data, pruned)
		
		def plotEvC(master, data, pruned):
			
			data = data.groupby("Output")
			attack = data.get_group("Attack")			
			normal = data.get_group("Normal")
			ax = attack['Energy_consumption']
			ay = attack['Cost']
			nx = normal['Energy_consumption']
			ny = normal['Cost']
			
			title = "Energy Consumption vs Cost"
			if (pruned == True):
				title += " (Pruned)"
				
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)
			#plot.set_title(title)		
			
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_ylabel('Cost', fontsize=12, color='#BDBDBD')
			plot.set_xlabel('Energy Consumption', fontsize=12, color='#BDBDBD')		
			plot.tick_params(axis='x', colors='#BDBDBD')
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.scatter(nx, ny, label = "Normal")
			plot.scatter(ax, ay, label = "Attack")
			plot.legend()			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")
		
		def plotE(master, data, pruned):
			
			data = data.groupby("Output")
			attack = data.get_group("Attack")			
			normal = data.get_group("Normal")
			ax = attack['Energy_consumption']
			ay = attack['Cost']
			nx = normal['Energy_consumption']
			ny = normal['Cost']
			
			title = "Energy Consumption Frequency"
			if (pruned == True):
				title += " (Pruned)"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)
			plot.set_title(title)
			
			
			
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_ylabel('Frequency', fontsize=12, color='#BDBDBD')
			plot.set_xlabel('Energy Consumption', fontsize=12, color='#BDBDBD')		
			plot.tick_params(axis='x', colors='#BDBDBD')
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.hist(nx, alpha = 0.5, label = "Normal", edgecolor = 'cyan')
			plot.hist(ax, alpha = 0.5, label = "Attack", edgecolor = 'orange')
			plot.legend()			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")			
		
		def plotC(master, data, pruned):
			
			data = data.groupby("Output")
			attack = data.get_group("Attack")			
			normal = data.get_group("Normal")
			ax = attack['Energy_consumption']
			ay = attack['Cost']
			nx = normal['Energy_consumption']
			ny = normal['Cost']
			
			title = "Cost Frequency"
			if (pruned == True):
				title += " (Pruned)"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)
			plot.set_title(title)
			plot.set_ylabel('Frequency')
			plot.set_xlabel('Cost')				
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_ylabel('Frequency', fontsize=12, color='#BDBDBD')
			plot.set_xlabel('Cost', fontsize=12, color='#BDBDBD')		
			plot.tick_params(axis='x', colors='#BDBDBD')
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.hist(ny, alpha = 0.5, label = "Normal", edgecolor = 'cyan')
			plot.hist(ay, alpha = 0.5, label = "Attack", edgecolor = 'orange')
			plot.legend()			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")
		
		def plotRatio(master, data, pruned):
			
			data = data.groupby("Output")
			attack = data.get_group("Attack")			
			normal = data.get_group("Normal")
			ax = attack['Energy_consumption']
			ay = attack['Cost']
			nx = normal['Energy_consumption']
			ny = normal['Cost']
			a = ax / ay
			n = nx / ny
			
			title = "Energy Consumption to Cost Ratio Frequency"
			if (pruned == True):
				title += " (Pruned)"
			
			#Figure creation
			figure = Figure()
			plot = figure.add_subplot(111)
			plot.set_title(title)
			plot.set_ylabel('Frequency')
			plot.set_xlabel('Cost')				
			#Plot Formatting
			plot.set_facecolor('#E0E0E0')
			figure.patch.set_color('#424242')
			plot.spines['bottom'].set_color('#BDBDBD')
			plot.spines['top'].set_color('#BDBDBD')
			plot.spines['right'].set_color('#BDBDBD')
			plot.spines['left'].set_color('#BDBDBD')
			plot.set_title(title, fontsize=14, color='#BDBDBD')
			plot.set_ylabel('Frequency', fontsize=12, color='#BDBDBD')
			plot.set_xlabel('Energy Consumption to Cost Ratio', fontsize=12, color='#BDBDBD')		
			plot.tick_params(axis='x', colors='#BDBDBD')
			plot.tick_params(axis='y', colors='#BDBDBD')
			mpl.rcParams['font.sans-serif'] = "Arial"
			mpl.rcParams['font.family'] = "sans-serif"	
			
			#Plot graph
			plot.hist(n, alpha = 0.5, label = "Normal", edgecolor = 'cyan')
			plot.hist(a, alpha = 0.5, label = "Attack", edgecolor = 'orange')
			plot.legend()			
			
			#Place graph
			canvas = FigureCanvasTkAgg(figure, master)
			canvas.draw()
			canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")		
			toolbarFrame = Frame(master=master)
			toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
			toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
			toolbar.config(background = "#424242")
			toolbar._message_label.config(background='#424242')
			
			for button in toolbar.winfo_children():
				button.config(background = '#E0E0E0', relief = "solid", bd = 1, )
			toolbar.update()			
			canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 2, sticky="NSEW")		
		

		
		#########
		# ROW 0 #
		#########
		fileLabel = CustomLabel(l_frame, text = "Load File", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		fileLabel.grid(row = 0, column = 0, columnspan = 2, sticky = "NSEW", padx = (0, 5))		
		
		optLabel = CustomLabel(l_frame, text = "File Options", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		optLabel.grid(row = 0, column = 3, columnspan = 1, sticky = "NSEW", padx = (5, 0))			
		
		
		
		#########
		# ROW 1 #
		#########
		#File Frame
		fileFrame = tk.Frame(l_frame, background = "#009699")
		fileFrame.grid(row = 1, column = 0, rowspan = 2, columnspan = 3, sticky = "NSEW", padx=(0,5))
		for row in range(1):
			fileFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			fileFrame.grid_columnconfigure(col, weight = 1)		
		
		fileFrame2 = tk.Frame(fileFrame, background = "#212121")
		fileFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			fileFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			fileFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Entry
		fname = tk.StringVar()
		fname.set("FDIdataset.csv")
		E1 = tk.Entry(fileFrame2, textvariable = fname, justify = "center", background = '#424242', foreground = '#757575', font = ('Arial', 12, 'bold'), borderwidth = 6, relief = "flat")
		E1.bind("<FocusIn>", handle_focus_in)
		E1.bind("<Return>", handle_return)
		E1.grid(row=0,column=0, columnspan = 2, sticky="NSEW", padx=(4,4), pady=(4, 4))			
		
		#Browse
		browse = CustomButton(fileFrame2, text = "Browse", command = load_file, font = ("Arial", 12), activebackground = "#616161")
		browse.grid(row = 1, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (0, 4))			
		
		
		#Options Frame
		optFrame = tk.Frame(l_frame, background = "#009699")
		optFrame.grid(row = 1, column = 3, rowspan = 2, columnspan = 1, sticky = "NSEW", padx=(5,0))
		for row in range(1):
			optFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			optFrame.grid_columnconfigure(col, weight = 1)		
		
		optFrame2 = tk.Frame(optFrame, background = "#212121")
		optFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			optFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			optFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Show Dataset Stats
		stats = CustomButton(optFrame2, text = "Show Dataset Stats", command = showStats, font = ("Arial", 12), activebackground = "#616161")
		stats.grid(row = 0, column = 0, sticky = "NSEW", padx = (4, 4), pady = (4, 2))		
		
		#Prune Duplicates
		prune = IntVar()
		prune_check = CustomCheckbutton(optFrame2, variable = prune, text = "Prune Duplicates")
		prune_check.grid(row=1,column=0, sticky="NSEW", padx=(4,4), pady=(2, 4))		
		
		
		
		#########
		# ROW 3 #
		#########		
		axisLabel = CustomLabel(l_frame, text = "Axis Graphs", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		axisLabel.grid(row = 3, column = 0, columnspan = 2, sticky = "NSEW", padx = (0, 5))		
		
		optLabel = CustomLabel(l_frame, text = "Random Forest Prediction Graphs", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		optLabel.grid(row = 3, column = 2, columnspan = 2, sticky = "NSEW", padx = (5, 0))		
		
		
		
		#########
		# ROW 4 #
		#########		
		#Axis Plot Frame
		axisFrame = tk.Frame(l_frame, background = "#009699")
		axisFrame.grid(row = 4, column = 0, rowspan = 1, columnspan = 2, sticky = "NSEW", padx=(0,5))
		for row in range(1):
			axisFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			axisFrame.grid_columnconfigure(col, weight = 1)		
		
		axisFrame2 = tk.Frame(axisFrame, background = "#212121")
		axisFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			axisFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			axisFrame2.grid_columnconfigure(col, weight = 1)			
		
		#Axis Combobox
		plotChoice = StringVar() 
		plotSel = ttk.Combobox(axisFrame2, state="readonly", justify = 'center')
		
		plotSel['values'] = (' Energy vs Cost',  
					     ' Energy Consumption',
					     ' Cost',
					     ' Energy to Cost Ratio') 
		plotSel.grid(row=0,column=0, sticky=NSEW, padx=(4,2), pady=(4,4))
		plotSel.current(0) 		
	
		#Show Plot button
		plot_button = CustomButton(axisFrame2, text = "Show Plot", command = lambda: plotSelection(r_frame2), activebackground = "#616161")
		plot_button.grid(row = 0, column = 1, columnspan = 1, sticky=NSEW, padx=(2,4), pady=(4,4))			
		
		
		#Random Forest Frame
		rfFrame = tk.Frame(l_frame, background = "#009699")
		rfFrame.grid(row = 4, column = 2, rowspan = 1, columnspan = 2, sticky = "NSEW", padx=(5,0))
		for row in range(1):
			rfFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			rfFrame.grid_columnconfigure(col, weight = 1)		
		
		rfFrame2 = tk.Frame(rfFrame, background = "#212121")
		rfFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			rfFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			rfFrame2.grid_columnconfigure(col, weight = 1)				
		
		#Random Forest Combobox
		plotChoice2 = StringVar() 
		rfChoice = ttk.Combobox(rfFrame2, state="readonly", justify = 'center') 
		rfChoice['values'] = (' FDI Training Set', ' FDI Test Set') 
		rfChoice.grid(row=0,column=0, sticky=NSEW, padx=(4,2), pady=(4,4))
		rfChoice.current(0)		
		
		#Random Forest Plot Button
		B2 = CustomButton(rfFrame2, text = "Show Plot", command = lambda : rfClassify(r_frame2), activebackground = "#616161")
		B2.grid(row=0, column=1, columnspan = 1, sticky=NSEW, padx=(2,4), pady=(4,4))		
		
		
		
		#########
		# ROW 5 #
		#########
		yearLabel = CustomLabel(l_frame, text = "Year Graphs", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		yearLabel.grid(row = 5, column = 0, columnspan = 1, sticky = "NSEW", padx = (0, 5))		
		
		monthLabel = CustomLabel(l_frame, text = "Month Graph", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		monthLabel.grid(row = 5, column = 1, columnspan = 1, sticky = "NSEW", padx = (5, 0))			
		
		dayLabel = CustomLabel(l_frame, text = "Day Graph", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		dayLabel.grid(row = 5, column = 2, columnspan = 2, sticky = "NSEW", padx = (5, 0))		
		
		
		
		#########
		# ROW 6 #
		#########		
		#Year Frame
		yearFrame = tk.Frame(l_frame, background = "#009699")
		yearFrame.grid(row = 6, column = 0, rowspan = 2, columnspan = 1, sticky = "NSEW", padx=(0,5))
		for row in range(1):
			yearFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			yearFrame.grid_columnconfigure(col, weight = 1)		
		
		yearFrame2 = tk.Frame(yearFrame, background = "#212121")
		yearFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			yearFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			yearFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Plot Year
		y_button = CustomButton(yearFrame2, text = "Plot Year", font = ("Arial", 12), activebackground = "#616161", command = lambda: plotYear(r_frame2))
		y_button.grid(row = 0, column = 0, columnspan = 1, sticky="NSEW", padx=(4,4), pady=(4, 2))				
		
		#Plot Hour
		h_button = CustomButton(yearFrame2, text = "Plot Hour Average", font = ("Arial", 12), activebackground = "#616161", command = lambda: plotHour(r_frame2))
		h_button.grid(row = 1, column = 0, columnspan = 1, sticky="NSEW", padx=(4,4), pady=(2, 4))			
		
		
		#Month Frame
		monthFrame = tk.Frame(l_frame, background = "#009699")
		monthFrame.grid(row = 6, column = 1, rowspan = 2, columnspan = 1, sticky = "NSEW", padx=(5,5))
		for row in range(1):
			monthFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			monthFrame.grid_columnconfigure(col, weight = 1)		
		
		monthFrame2 = tk.Frame(monthFrame, background = "#212121")
		monthFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			monthFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			monthFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Month Choice
		data = pd.read_csv(E1.get())
		vals = list(np.unique(pd.to_datetime(data['Date'].unique()).month))
		for i in range(len(vals)):
			vals[i] = str(calendar.month_name[vals[i]])
		
		m_choice = ttk.Combobox(monthFrame2, width = 20, state="readonly", justify = 'center')
		m_choice['values'] = vals
		m_choice.grid(row=0,column=0, sticky=NSEW, padx=(4,4), pady=(4,2))	
		m_choice.current(0)
		
		#Plot Month
		m_button = CustomButton(monthFrame2, text = "Plot Month", font = ("Arial", 12), activebackground = "#616161", command = lambda: plotMonth(r_frame2, m_choice.current()+1))
		m_button.grid(row = 1, column = 0, columnspan = 1, sticky="NSEW", padx=(4,4), pady=(2, 4))				
		
		
		#Day Frame
		dayFrame = tk.Frame(l_frame, background = "#009699")
		dayFrame.grid(row = 6, column = 2, rowspan = 2, columnspan = 2, sticky = "NSEW", padx=(5,0))
		for row in range(1):
			dayFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			dayFrame.grid_columnconfigure(col, weight = 1)		
		
		dayFrame2 = tk.Frame(dayFrame, background = "#212121")
		dayFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			dayFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			dayFrame2.grid_columnconfigure(col, weight = 1)			
		
		def setDays():
			month = dm_choice.current() + 1
			vals = []
			vals = calendar.monthcalendar(2010, month)
			vals = list(np.unique(np.concatenate(vals).flat))
			if (vals[0] == 0):
				vals = list(np.delete(vals, 0))
			d_choice['values'] = vals
		
		#Month Choice for Day
		dm_choice = ttk.Combobox(dayFrame2, width = 20, state="readonly", postcommand = lambda: d_choice.current(0), justify = 'center')
		dm_choice['values'] = vals
		dm_choice.grid(row=0,column=0, sticky=NSEW, padx=(4,2), pady=(4,2))	
		dm_choice.current(0)
		
		#Day Choice
		d_choice = ttk.Combobox(dayFrame2, width = 20, state="readonly", postcommand = setDays, justify = 'center')
		d_choice['values'] = (['1'])
		d_choice.grid(row=0,column=1, sticky=NSEW, padx=(2,4), pady=(4,2))	
		d_choice.current(0)			
		
		#Plot Day
		d_button = CustomButton(dayFrame2, text = "Plot Day", font = ("Arial", 12), activebackground = "#616161", command = lambda: plotDay(r_frame2, dm_choice.current()+1, d_choice.get()))
		d_button.grid(row = 1, column = 0, columnspan = 2, sticky="NSEW", padx=(4,4), pady=(2, 4))		
		
		
		
		###############################
		# RIGHT FRAME GRID FORMATTING #
		###############################
		for col in range(1):
			r_frame.grid_columnconfigure(col, weight=1)
	
		for row in range(1):
			r_frame.grid_rowconfigure(row, weight = 1)
		
		r_frame.grid_propagate(0)
		
		r_frame2 = tk.Frame(r_frame, background = "#212121")
		r_frame2.grid(row = 0, column = 0, sticky = "NSEW", padx=(2,12), pady=2)
		for row in range(1):
			r_frame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			r_frame2.grid_columnconfigure(col, weight = 1)		
		

	def load_file(self):
		self.fname = filedialog.askopenfilename()	


"""
###########
# TAB TWO #
###########
"""
class TabTwo(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg="#212121", width = 1600, height = 700)
		self.grid(row = 0, column = 0, sticky = "NSEW")
		self.grid_propagate(False)
		
		self.grid_rowconfigure(0, minsize=233)
		self.grid_columnconfigure(0, minsize=200)		
		
		
		self.controller = controller
		
		#Frame formatting
		l_frame = tk.Frame(self, bg='#212121', width = 800, height = 700)
		#m_frame = Frame(root, bg='red', width = 850, height = 700)
		r_frame = tk.Frame(self, bg='#BF4300', width = 800, height = 700)
		#212121
		
		l_frame.grid(row = 0, column = 0, sticky = "NSEW", padx=[0, 5])
		l_frame.grid_propagate(False)
		
		r_frame.grid(row = 0, column = 1, sticky = "NSEW", padx=[5, 0])
		r_frame.grid_propagate(False)			
		
		##############################
		# LEFT FRAME GRID FORMATTING #
		##############################
		for col in range(2):
			l_frame.grid_columnconfigure(col, weight=1)
		
		for row in range(19):
			#l_frame.grid_rowconfigure(row, minsize=60)
			l_frame.grid_rowconfigure(row, weight=1)
			
		
		#Functions
		def load_file():
			file = filedialog.askopenfilename(filetypes = [('CSV files', '.csv')], title = "Dataset Selection", initialdir = os.getcwd())
			fname.set(os.path.split(file)[1])
			#print(fname.get())
	
		def handle_focus_in(_):
			E1.select_range(0, END)
			E1.config(fg='#E0E0E0', selectbackground = "#BF4300", highlightbackground = '#616161', highlightcolor = '#BF4300', highlightthickness = 1, borderwidth = 5)		
	
		def handle_return(_):
			fname.set(E1.get())
			#print(fname.get())
			
		
		
		#########
		# ROW 0 #
		#########
		fileLabel = CustomLabel(l_frame, text = "Load File", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		fileLabel.grid(row = 0, column = 0, columnspan = 2, sticky = "NSEW", padx = (0, 5))		
		
		
		
		#########
		# ROW 1 #
		#########
		#File Frame
		fileFrame = tk.Frame(l_frame, background = "#BF4300")
		fileFrame.grid(row = 1, column = 0, rowspan = 1, columnspan = 2, sticky = "NSEW", padx=(0,0))
		for row in range(1):
			fileFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			fileFrame.grid_columnconfigure(col, weight = 1)		
		
		fileFrame2 = tk.Frame(fileFrame, background = "#212121")
		fileFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			fileFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			fileFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Entry
		fname = tk.StringVar()
		fname.set("FDIdataset.csv")
		E1 = tk.Entry(fileFrame2, textvariable = fname, justify = "center", background = '#424242', foreground = '#757575', font = ('Arial', 12, 'bold'), borderwidth = 6, relief = "flat")
		E1.bind("<FocusIn>", handle_focus_in)
		E1.bind("<Return>", handle_return)
		E1.grid(row=0,column=0, columnspan = 1, sticky="NSEW", padx=(4,4), pady=(4, 4))			
		
		#Browse
		browse = CustomButton(fileFrame2, text = "Browse", command = load_file, font = ("Arial", 12), activebackground = "#616161")
		browse.grid(row = 0, column = 1, columnspan = 1, sticky = "NSEW", padx = (0, 4), pady = (4, 4))					
		
		
		
		#########
		# ROW 2 #
		#########
		paramLabel = CustomLabel(l_frame, text = "Parameter", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		paramLabel.grid(row = 2, column = 0, columnspan = 1, sticky = "NSEW", padx = (0, 5))		
		
		valLabel = CustomLabel(l_frame, text = "Value", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		valLabel.grid(row = 2, column = 1, columnspan = 1, sticky = "NSEW", padx = (5, 0))			
		
		
		#########
		# ROW 3 #
		#########
		#Parameter Frame
		paramFrame = tk.Frame(l_frame, background = "#BF4300")
		paramFrame.grid(row = 3, column = 0, rowspan = 7, columnspan = 2, sticky = "NSEW", padx=(0,0))
		for row in range(1):
			paramFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			paramFrame.grid_columnconfigure(col, weight = 1)		
		
		paramFrame2 = tk.Frame(paramFrame, background = "#212121")
		paramFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(7):
			paramFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(3):
			paramFrame2.grid_columnconfigure(col, weight = 1)		
		paramFrame2.grid_columnconfigure(1, weight = 0)
		
		#############
		# SEPERATOR #
		#############
		sep = tk.Frame(paramFrame2, width = 2, background = "#BF4300")
		sep.grid(row=0, column=1, rowspan = 20, sticky=NS, padx=(0,0), pady=(4,4))		
		
		
		
		##############
		# PARAMETERS #
		##############	
		#Start
		start_label = CustomLabel(paramFrame2, text="Start Time")
		start_label.grid(row = 0, column = 0, sticky = "NSEW", padx = (4, 4), pady = (4, 2))
		
		start_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		temp = np.arange(0, 1431, 10)	
		start_list['values'] = list(temp)
		start_list.grid(row=0,column=2, sticky = "NSEW", padx = (4, 4), pady = (4, 2))
		start_list.current(0) 				
		
		
		#Stop
		stop_label = CustomLabel(paramFrame2, text="Stop Time")
		stop_label.grid(row = 1, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		stop_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		temp = np.arange(0, 1431, 10)		
		stop_list['values'] = list(temp)		
		stop_list.grid(row=1,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		stop_list.current(80)
		
		
		#Attack Percent
		attack_percent_label = CustomLabel(paramFrame2, text="Attack Percent Increase")
		attack_percent_label.grid(row = 2, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		attack_percent_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		temp = np.arange(5, 101, 5)
		attack_percent_list['values'] = list(temp)
		attack_percent_list.grid(row=2,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		attack_percent_list.current(4) 	
		
		
		#Day Percent
		day_percent_label = CustomLabel(paramFrame2, text="Percent Injection Days")
		day_percent_label.grid(row = 3, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		day_percent_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		temp = np.arange(10, 101, 10)
		day_percent_list['values'] = list(temp)
		day_percent_list.grid(row=3,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		day_percent_list.current(7) 			
		
		
		#Weight
		weight_label = CustomLabel(paramFrame2, text="Weight")
		weight_label.grid(row = 4, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		weight_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		weight_list['values'] = (' 1', ' 2',' 3') 
		weight_list.grid(row=4,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		weight_list.current(1) 				
		
		#Force Attack
		force_attack_label = CustomLabel(paramFrame2, text="Force Attacks")
		force_attack_label.grid(row = 5, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		force_attack_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		force_attack_list['values'] = (' True', ' False') 
		force_attack_list.grid(row=5,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		force_attack_list.current(0) 			
		
		
		#Seed
		seed_label = CustomLabel(paramFrame2, text="Seed")
		seed_label.grid(row = 6, column = 0, sticky = "NSEW", padx = (4, 4), pady = (2, 4))
		
		seed_list = ttk.Combobox(paramFrame2, width = 20, state="readonly", justify = 'center')
		seed_list['values'] = (' 0',' 1', ' 2',' 3')
		seed_list.grid(row=6,column=2, sticky = "NSEW", padx = (4, 4), pady = (2, 4))
		seed_list.current(0) 			
		
		
		
		#########
		# ROW 7 #
		#########		
		eqLabel = CustomLabel(l_frame, text = "Curve Fit Equations", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		eqLabel.grid(row = 10, column = 0, columnspan = 2, sticky = "NSEW", padx = (0, 0))		
		
		
		#########
		# ROW 8 #
		#########
		#Equation Frame
		eqFrame = tk.Frame(l_frame, background = "#BF4300")
		eqFrame.grid(row = 11, column = 0, rowspan = 7, columnspan = 2, sticky = "NSEW", padx=(0,0))
		for row in range(1):
			eqFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			eqFrame.grid_columnconfigure(col, weight = 1)		
		
		eqFrame2 = tk.Frame(eqFrame, background = "#212121")
		eqFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			eqFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			eqFrame2.grid_columnconfigure(col, weight = 1)	
		eqFrame2.grid_propagate(False)
		
		
		##########
		# ROW 18 #
		##########		
		#Run Frame
		runFrame = tk.Frame(l_frame, background = "#BF4300")
		runFrame.grid(row = 18, column = 0, rowspan = 7, columnspan = 2, sticky = "NSEW", padx=(0,0), pady=(10,0))
		for row in range(1):
			runFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			runFrame.grid_columnconfigure(col, weight = 1)		
		
		runFrame2 = tk.Frame(runFrame, background = "#212121")
		runFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			runFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			runFrame2.grid_columnconfigure(col, weight = 1)			
		
		#Run
		run_inject = CustomButton(runFrame2, text = "Run", font = ("Arial", 12, 'bold'), activebackground = "#616161", command = lambda : threading.Thread(
		        target = AttackInject(E1.get(), r_frame2, seed_list.get(), start_list.get(), stop_list.get(), attack_percent_list.get(), day_percent_list.get(), weight_list.get(), force_attack_list.get(), eqFrame2)).start())
		run_inject.grid(row = 0, column = 0, columnspan = 1, sticky = "NSEW", padx = (4, 4), pady = (4, 4))		
		
		
		
		
		###############################
		# RIGHT FRAME GRID FORMATTING #
		###############################
		for col in range(1):
			r_frame.grid_columnconfigure(col, weight=1)
	
		for row in range(1):
			r_frame.grid_rowconfigure(row, weight = 1)
		
		r_frame.grid_propagate(0)
		
		r_frame2 = tk.Frame(r_frame, background = "#212121")
		r_frame2.grid(row = 0, column = 0, sticky = "NSEW", padx=(2,12), pady=2)
		for row in range(1):
			r_frame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			r_frame2.grid_columnconfigure(col, weight = 1)			

"""
#############
# TAB THREE #
#############
"""
class TabThree(tk.Frame):
	
	
	
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent, bg="#212121", width = 1600, height = 700)
		
		self.grid(row = 0, column = 0, sticky = "NSEW")
		self.grid_propagate(False)
		self.grid_rowconfigure(0, minsize=233)
		self.grid_columnconfigure(0, minsize=200)
		self.controller = controller
		
		#Frame formatting
		l_frame = tk.Frame(self, bg='#212121', width = 800, height = 700)
		r_frame = tk.Frame(self, bg='#339900', width = 800, height = 700)
		
		l_frame.grid(row = 0, column = 0, sticky = "NSEW", padx=[0, 5])
		l_frame.grid_propagate(False)
		
		r_frame.grid(row = 0, column = 1, sticky = "NSEW", padx=[5, 0])
		r_frame.grid_propagate(False)			
		
		
		##############################
		# LEFT FRAME GRID FORMATTING #
		##############################
		for col in range(5):
			l_frame.grid_columnconfigure(col, weight=1)
		for row in range(20):
			#l_frame.grid_rowconfigure(row, minsize=60)
			l_frame.grid_rowconfigure(row, weight = 1)
			
		l_frame.grid_propagate(0)
		
		#Functions
		def load_file():
			file = filedialog.askopenfilename(filetypes = [('CSV files', '.csv')], title = "Dataset Selection", initialdir = os.getcwd())
			fname.set(os.path.split(file)[1])
			#print(fname.get())
	
		def handle_focus_in(_):
			E1.select_range(0, END)
			E1.config(fg='#E0E0E0', selectbackground = "#339900", highlightbackground = '#616161', highlightcolor = '#339900', highlightthickness = 1, borderwidth = 5)	
		
		def handle_return(_):
			fname.set(E1.get())
			#print(fname.get())
		
		
		
		#########
		# ROW 0 #
		#########
		fileLabel = CustomLabel(l_frame, text = "Load File", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		fileLabel.grid(row = 0, column = 0, columnspan = 2, sticky = "NSEW", padx = (0, 5))		
		
		optLabel = CustomLabel(l_frame, text = "Meta Options", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		optLabel.grid(row = 0, column = 3, columnspan = 1, sticky = "NSEW", padx = (5, 0))			
		
		
		
		#########
		# ROW 1 #
		#########
		#File Frame
		fileFrame = tk.Frame(l_frame, background = "#339900")
		fileFrame.grid(row = 1, column = 0, rowspan = 2, columnspan = 2, sticky = "NSEW", padx=(0,5))
		for row in range(1):
			fileFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			fileFrame.grid_columnconfigure(col, weight = 1)		
		
		fileFrame2 = tk.Frame(fileFrame, background = "#212121")
		fileFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			fileFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(2):
			fileFrame2.grid_columnconfigure(col, weight = 1)		
		
		#Entry
		fname = tk.StringVar()
		fname.set("FDIdataset.csv")
		E1 = tk.Entry(fileFrame2, textvariable = fname, justify = "center", background = '#424242', foreground = '#757575', font = ('Arial', 12, 'bold'), borderwidth = 6, relief = "flat")
		E1.bind("<FocusIn>", handle_focus_in)
		E1.bind("<Return>", handle_return)
		E1.grid(row=0,column=0, columnspan = 2, sticky="NSEW", padx=(4,4), pady=(4, 4))			
		
		#Browse
		browse = CustomButton(fileFrame2, text = "Browse", command = load_file, font = ("Arial", 12), activebackground = "#616161")
		browse.grid(row = 1, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (0, 4))			
		
		
		
		"""
		####################################
		# Random Forest Control Parameters #
		####################################
		"""			
		#Options Frame
		optFrame = tk.Frame(l_frame, background = "#339900")
		optFrame.grid(row = 1, column = 2, rowspan = 2, columnspan = 3, sticky = "NSEW", padx=(5,0))
		for row in range(1):
			optFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			optFrame.grid_columnconfigure(col, weight = 1)		
		
		optFrame2 = tk.Frame(optFrame, background = "#212121")
		optFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(2):
			optFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(3):
			optFrame2.grid_columnconfigure(col, weight = 1)
		optFrame2.grid_propagate(0)
		
		#Seed
		seedChoice = StringVar()
		seed_label = CustomLabel(optFrame2, text="Seed")
		seed_label.grid(row = 0, column = 0, sticky = "NSEW", padx = (4, 2), pady = (4, 2))
		seed_list = ttk.Combobox(optFrame2, width = 20, textvariable = seedChoice, state="readonly", justify = 'center')
		seed_list['values'] = (' 0', ' 1', ' 2',' 3') 
		seed_list.grid(row=1,column=0, columnspan = 1, sticky = "NSEW", padx = (4, 2), pady = (2, 4))
		seed_list.current(0) 		
		
		#Training Percent
		trainChoice = StringVar()
		train_label = CustomLabel(optFrame2, text="Training Percent")
		train_label.grid(row = 0, column = 1, sticky = "NSEW", padx = (2, 2), pady = (4, 2))
		train_list = ttk.Combobox(optFrame2, width = 20, textvariable = trainChoice, state="readonly", justify = 'center')
		train_list['values'] = (' 60', ' 70', ' 75',' 80') 
		train_list.grid(row=1,column=1, columnspan = 1, sticky = "NSEW", padx = (2, 2), pady = (2, 4))
		train_list.current(0)		
		
		#Number of Jobs
		n_jobsChoice = StringVar()
		n_jobs_label = CustomLabel(optFrame2, text="Number of Jobs")
		n_jobs_label.grid(row = 0, column = 2, sticky = "NSEW", padx = (2, 4), pady = (4, 2))
		n_jobs_list = ttk.Combobox(optFrame2, width = 20, textvariable = n_jobsChoice, state="readonly", justify = 'center')
		n_jobs_list['values'] = (' 2', ' 4', ' 6',' 8', ' 10') 
		n_jobs_list.grid(row=1,column=2, columnspan = 1, sticky = "NSEW", padx = (2, 4), pady = (2, 4))
		n_jobs_list.current(2)
		
		
		
		#########
		# ROW 3 #
		#########
		paramLabel = CustomLabel(l_frame, text = "Parameter Grid Option", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		paramLabel.grid(row = 3, column = 0, columnspan = 1, sticky = "NSEW", padx = (0, 5))		
		
		valLabel = CustomLabel(l_frame, text = "Value", background = "#212121", anchor = 'sw', font = ['Arial', 14, 'normal'], padx = 10)
		valLabel.grid(row = 3, column = 1, columnspan = 4, sticky = "NSEW", padx = (5, 0))			
		
		
		
		#########
		# ROW 4 #
		#########
		#Parameter Frame
		paramFrame = tk.Frame(l_frame, background = "#339900")
		paramFrame.grid(row = 4, column = 0, rowspan = 15, columnspan = 5, sticky = "NSEW", padx=(0,0))
		for row in range(1):
			paramFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			paramFrame.grid_columnconfigure(col, weight = 1)		
		
		paramFrame2 = tk.Frame(paramFrame, background = "#212121")
		paramFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(7):
			paramFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(5):
			paramFrame2.grid_columnconfigure(col, weight = 1)
		paramFrame2.grid_columnconfigure(2, weight = 0)
		paramFrame.grid_propagate(0)
		paramFrame2.grid_propagate(0)
		
		#############
		# SEPERATOR #
		#############
		sep = tk.Frame(paramFrame2, width = 2, background = "#339900")
		sep.grid(row=0, column=2, rowspan = 7, sticky=NS, padx=(0,0), pady=(4,4))
		
		
		"""
		#################################
		# Random Forest Hyperparameters #
		#################################
		"""
		# Number of Estimators (trees in forest)
		#n_estimators = [int(x) for x in np.linspace(start = 100,stop = 1000, num = 5)] # 10 was best out of [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000]
		startChoice = StringVar()
		stopChoice = StringVar()
		numChoice = StringVar()
		
		n_est_label = CustomLabel(paramFrame2, text="Number of Estimators\n(start, stop, number)")
		n_est_label.grid(row = 0, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (4, 2))
		
		temp = list(np.arange(100, 1001, 100))
		start_list = ttk.Combobox(paramFrame2, textvariable = startChoice, state="readonly", justify = 'center', width = 10)
		start_list['values'] = temp
		start_list.grid(row=0,column=3, sticky = "NSEW", padx = (4, 2), pady = (4, 2))
		start_list.current(0) 		
		
		stop_list = ttk.Combobox(paramFrame2, textvariable = stopChoice, state="readonly", justify = 'center', width = 10)
		stop_list['values'] = temp
		stop_list.grid(row=0,column=4, sticky = "NSEW", padx = (2, 2), pady = (4, 2))
		stop_list.current(1)
		
		num_list = ttk.Combobox(paramFrame2, textvariable = numChoice, state="readonly", justify = 'center', width = 10)
		num_list['values'] = (' 1', ' 2', ' 5',' 10',' 25') 
		num_list.grid(row=0,column=5, sticky = "NSEW", padx = (2, 4), pady = (4, 2))
		num_list.current(0)
		
		#Max Features
		# Number of features to consider at every split - #max_features = ['auto'] # auto > sqrt and log2
		max_featuresChoice = StringVar()
		max_features_label = CustomLabel(paramFrame2, text="Max Features")
		max_features_label.grid(row = 1, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
	
		max_features_list = ttk.Combobox(paramFrame2, textvariable = max_featuresChoice, state="readonly", justify = 'center')
		max_features_list['values'] = (' auto', ' sqrt', ' log2',' None') 
		max_features_list.grid(row=1,column=3, columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		max_features_list.current(0) 		
		
		
		#Criterion
		# Measures quality of the split - #criterion= ['entropy','gini'] # 'entropy' better than 'gini index'
		critChoice = StringVar()
		crit_label = CustomLabel(paramFrame2, text="Criterion")
		crit_label.grid(row = 2, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		crit_list = ttk.Combobox(paramFrame2, textvariable = critChoice, state="readonly", justify = 'center')
		crit_list['values'] = (' entropy, gini', ' entropy', ' gini') 
		crit_list.grid(row=2,column=3, columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		crit_list.current(0) 		
		
		#Max Depth
		# Maximum number of levels in tree - #max_depth = [10] #10 best out of [2,5,10,15,20,50]
		max_depthChoice = StringVar()
		max_depth_label = CustomLabel(paramFrame2, text="Max Depth")
		max_depth_label.grid(row = 3, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		max_depth_list = ttk.Combobox(paramFrame2, textvariable = max_depthChoice, state="readonly", justify = 'center')
		max_depth_list['values'] = (' 2', ' 5', ' 10', ' 15', ' 20', ' 50') 
		max_depth_list.grid(row=3,column=3, columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		max_depth_list.current(2) 		
		
		#Min Samples Split
		# Minimum number of samples required to split a node - #min_samples_split = [10] # 10 best out of [2,5,10,20]
		min_ssplitChoice = StringVar()
		min_ssplit_label = CustomLabel(paramFrame2, text="Min Samples Split")
		min_ssplit_label.grid(row = 4, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		min_ssplit_list = ttk.Combobox(paramFrame2, textvariable = min_ssplitChoice, state="readonly", justify = 'center')
		min_ssplit_list['values'] = (' 2', ' 5', ' 10', ' 20') 
		min_ssplit_list.grid(row=4,column=3 ,columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		min_ssplit_list.current(2)
		
		#Min Samples Leaf
		# minimum number of samples required at each leaf node - #min_samples_leaf = [15] # 15 best out of [5,10,15,20,25,30]
		min_sleafChoice = StringVar()
		min_sleaf_label = CustomLabel(paramFrame2, text="Min Samples Leaf")
		min_sleaf_label.grid(row = 5, column = 0, columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		min_sleaf_list = ttk.Combobox(paramFrame2, textvariable = min_sleafChoice, state="readonly", justify = 'center')
		min_sleaf_list['values'] = (' 2', ' 5', ' 10', ' 15', ' 25', ' 30') 
		min_sleaf_list.grid(row=5,column=3,columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		min_sleaf_list.current(3) 		
		
		#Bootstrap
		# Method of selecting samples for training each tree - #bootstrap = [True] # False was best out [True, False]
		bootstrapChoice = StringVar()
		bootstrap_label = CustomLabel(paramFrame2, text="Bootstrap")
		bootstrap_label.grid(row = 6, column = 0,  columnspan = 2, sticky = "NSEW", padx = (4, 4), pady = (2, 2))
		
		bootstrap_list = ttk.Combobox(paramFrame2, textvariable = bootstrapChoice, state="readonly", justify = 'center')
		bootstrap_list['values'] = (' True', ' False') 
		bootstrap_list.grid(row=6,column=3,columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (2, 4))
		bootstrap_list.current(1) 		
		
		
		
		#########
		# ROW 11 #
		#########
		#Run Frame
		runFrame = tk.Frame(l_frame, background = "#339900")
		runFrame.grid(row = 19, column = 0, rowspan = 1, columnspan = 5, sticky = "NSEW", padx=(0,0), pady =(10, 0))
		for row in range(1):
			runFrame.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			runFrame.grid_columnconfigure(col, weight = 1)		
		
		runFrame2 = tk.Frame(runFrame, background = "#212121")
		runFrame2.grid(row = 0, column = 0, sticky = "NSEW", padx=2, pady=2)
		for row in range(1):
			runFrame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			runFrame2.grid_columnconfigure(col, weight = 1)
		
		#Run
		run = CustomButton(runFrame2, text = "Run", font = ("Arial", 12, 'bold'), activebackground = "#616161", command = lambda : threading.Thread(
		        target = RunRandomForest(fname, r_frame2, seedChoice, trainChoice, n_jobsChoice, startChoice, stopChoice, numChoice, max_featuresChoice, critChoice, max_depthChoice,
		                                 min_ssplitChoice, min_sleafChoice, bootstrapChoice)).start())
		run.grid(row = 0, column = 0, columnspan = 3, sticky = "NSEW", padx = (4, 4), pady = (4, 4))
		
		
		###############################
		# RIGHT FRAME GRID FORMATTING #
		###############################
		for col in range(1):
			r_frame.grid_columnconfigure(col, weight=1)
	
		for row in range(1):
			r_frame.grid_rowconfigure(row, weight = 1)
		
		r_frame.grid_propagate(0)
		
		r_frame2 = tk.Frame(r_frame, background = "#212121")
		r_frame2.grid(row = 0, column = 0, sticky = "NSEW", padx=(2,12), pady=2)
		for row in range(1):
			r_frame2.grid_rowconfigure(row, weight = 1)
		for col in range(1):
			r_frame2.grid_columnconfigure(col, weight = 1)	
	

if __name__ == "__main__":

	
	app = App()
	app.mainloop()
	
