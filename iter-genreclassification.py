#!/usr/bin/python3

import math
import string
import os
import random
import sys
import matplotlib
import codecs
import unicodedata
from collections import defaultdict
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Calculate mean of the values  
def m(values):
	size = len(values)
	sum = 0.0
	for n in range(0, size):
		sum += values[n]
	return(sum / size)

# Calculate standard deviation  
def SD(values, mean):
	size = len(values)
	sum = 0.0
	for n in range(0, size):
		sum += math.sqrt((values[n] - mean)**2)
	return(math.sqrt((1.0/(size-1))*(sum/size)))

# Calculate standard error
def SE(values, mean):  
	sd = SD(values,mean)
	return(sd/math.sqrt(len(values)))




# generates k-fold cross validation indices
def k_fold_cross_validation(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.

	source: http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
	"""
	if randomise: from random import shuffle; X=list(X); shuffle(X)

	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

# gets number of lines in file
def file_len(fname):
	f = open(fname,'r')
	i = 0
	for i, l in enumerate(f):
		pass
	return i + 1

# stores 10-fold cross validation sets in train-[basename]-#.txt (#: 0-9) and
# develop-[basename]-#.txt
def generateKfold(trainingfilename,basename,kval):
	global develdocs, traindocs

	nrinstances = int(file_len(trainingfilename)/3)
	X = [i for i in range(nrinstances)]
	k = 0
	for training, validation in k_fold_cross_validation(X, K=kval):
		trainfile = open('tmp/train-' + basename+'-' + str(k) + '.txt','w')
		developfile = open('tmp/develop-' + basename+'-' + str(k) + '.txt','w')
		k += 1
		thefile = open(trainingfilename, 'r')
		linenr = 0
		for line in thefile:
			docnr = int(linenr / 3)
			linenr += 1
			if kval==1:
				developfile.write(line)
				trainfile.write(line)
			else:
				if docnr in validation:
					developfile.write(line)
				elif docnr in training:
					trainfile.write(line)
		thefile.close()

# reads the weight file
def readweights(filename):
	weights = []

	thefile = open(filename, 'r')
	for line in thefile:
		weights.append(float(line.strip()))
	return(weights)

# reads features from filename and reads the sorting indices from sorting
# returns the sorted names
def sortedfeatnames(filename,sorting):
	features = []
	features.append("IGNORED")

	thefile = codecs.open(filename, 'r','utf-8')
	for line in thefile:
		features.append(line.strip())

	os.system("grep \"^\*\" " + sorting + " | sed 's/^\* //g' > tmp/sorted-features.txt")

	sortedfeatures = []
	ranking = open('tmp/sorted-features.txt', 'r')
	for line in ranking:
		sortedfeatures.append(features[int(line.strip())])

	return(sortedfeatures)

# trains using tinyest
def train(trainfilename,l1=0,grafting_n=0,stdoutfile="",stderrfile=""):
	command = tinyestpath + "tinyest "
	
	if l1 > 0:
		command += "--l1 " + str(l1) + " "
	if grafting_n > 0:
		command += "--grafting " + str(grafting_n) + " "

	command += trainfilename + " "
	if stdoutfile != "":
		command += "1> " + stdoutfile + " "
	if stderrfile != "":
		command += "2> " + stderrfile
	os.system(command)


# runs a single multilevel test given all necessary weightfiles and the associated categories (cats)
# of the testfile with filename, returns the average performance
def multileveltest(filename,weightfiles,cats,outfilename='results/classification.txt'):
	
	weightslist = defaultdict(list)
	score = {}
	expscore = {}
	zc = {}
	convergenceError = {}
	prob = {}

	thefile = open(filename, 'r')
	outfile = open(outfilename, 'w')
	
	for i in range(0,len(weightfiles)):
		weightslist[cats[i]] = readweights(weightfiles[i])

	outfile.write('true class\t'+'assigned class\t' + 'correct')
	for cat in cats:
		outfile.write('\tprob-' + cat)
		convergenceError[cat]=False

	outfile.write('\n')

	totalcases = 0
	totalcorrect = 0

	for line in thefile:
		scores = line.strip().split(' ')
		if len(scores) > 2: # has scores
			totalcases = totalcases + 1
			trueclass = scores[0]
			nfeat = int(scores[1])
			for category in cats:
				score[category] = 0
				
			for i in range(0,nfeat):
				featnr = int(scores[i*2+2])
				featval = float(scores[i*2+3])

				for category in cats:
					weights = weightslist[category]
					if len(weights) > 0: # if convergence error, weights will be 0
						if featnr < len(weights): # test data might contan feature absent from training (then value == 0)
							score[category] += weights[featnr]*featval
					else:
						if not convergenceError[category]:
							print("Convergence error...")
							convergenceError[category]=True # holds for weight file, so for all cases
			
			curprob = 0
			assignedclass = "#NONE#"
			for category in cats:

				if score[category] > 100:
					prob[category] = 1
				else: 
					expscore[category] = math.exp(score[category])
					zc[category] = expscore[category] + math.exp(0)
					prob[category] = expscore[category] / zc[category]
				
				if convergenceError[category]: # if it failed to converge
					prob[category] = -100 # never the highest probability

				if prob[category] > curprob:
					assignedclass = category
					curprob = prob[category]
			
			correct = 0
			if (assignedclass == trueclass):
				correct = 1

			totalcorrect = totalcorrect + correct

			outfile.write(str(trueclass) + '\t' + str(assignedclass) + '\t' + str(correct) + '\t')
			for cat in cats:
				outfile.write('\t' + str(prob[cat]))
			outfile.write('\n')

	if (totalcases == 0):
		accuracy = -100
	else: 
		accuracy = (100.0 * float(totalcorrect)) / float(totalcases)

	outfile.write('\nperformance: ' + str(accuracy) + '%\n')
	#print('\nperformance: ' + str(accuracy) + '%\n')
	return(accuracy)


# runs a single twolevel test (correct vs. incorrect)
# used to determine the best L1 value for each genre
def runtest(filename,weights,outfilename='tmp/classification.txt'):
	thefile = open(filename, 'r')
	outfile = open(outfilename, 'w')
	outfile.write('true class\t'+'assigned class\t' + 'correct\t' +  'estimated probability\n')

	totalcases = 0
	totalcorrect = 0
	convergenceError = False
	for line in thefile:
		scores = line.strip().split(' ') 
		if len(scores) > 2: # has scores
			totalcases = totalcases + 1
			trueclass = int(scores[0]) # 1 or 0
			nfeat = int(scores[1])
			score = 0
			for i in range(0,nfeat):
				featnr = int(scores[i*2+2])
				featval = float(scores[i*2+3])

				if len(weights) > 0: # if convergence error, weights will be 0
					score += weights[featnr]*featval
				else:
					if not convergenceError:
						print("Convergence error...")
						convergenceError=True

			if score > 100: # e^100 / (e^100 + 1)
				prob = 1
			else: 
				expscore = math.exp(score)
				zc = expscore + math.exp(0)
				prob = expscore / zc

			if convergenceError:
				prob = -100

			if (prob == 0.5): # banker's rounding rounds 0.5 and -0.5 both to 0...
				assignedclass = 1
			elif (prob == -100):
				assignedclass = -100
			else: 
				assignedclass = int(round(prob))

			correct = 0
			if (assignedclass == trueclass):
				correct = 1

			totalcorrect = totalcorrect + correct

			outfile.write(str(trueclass) + '\t' + str(assignedclass) + '\t' + str(correct) + '\t' + str(prob) + '\n')

	if (totalcases == 0):
		accuracy = -100
	else:
		accuracy = (100.0 * float(totalcorrect)) / float(totalcases)

	outfile.write('\nperformance: ' + str(accuracy) + '%\n')

	if (convergenceError): # then all values are -100
		return(None)
	else:
		return(accuracy)

# iterates over all L1s to find the best one, for the binary classifier
def findBestL1(basename):
	trainingfile = "tmp/training-"+basename+".txt"
	print("2. Determining best L1 (based on 10-fold cross validation):")
	generateKfold(trainingfile,basename,10) # results in 10 train-[basename]-#.txt and develop-[basename]-#.txt files

	# l1s to test
	l1s = [0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]

	# direct output instead of buffered
	#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
	performance = []

	# train models for each l1 value and evaluate the
	# performance using 10-fold cross validation
	for l1 in range(0,len(l1s)): 
		accur = 0
		print("   Evaluating L1 " + str(l1s[l1]),end='')
		cnt = 0
		for k in range(0,10):
			print('.',end='')
			sys.stdout.flush() 
			train('tmp/train-' + basename + '-' + str(k) + '.txt',l1s[l1],0,'tmp/weights.txt','tmp/tmp.proc') # results in weights.txt file
			weights = readweights('tmp/weights.txt') # read trained weights
			
			a = runtest('tmp/develop-' + basename + '-' + str(k) + '.txt',weights,'tmp/classification.txt') # test on 10% of data 
			#print(a,end='')
			if a is not None:
				accur += a
				cnt += 1
		avgacc = accur/cnt

		print(": " + str(avgacc))
		performance.append(avgacc)

	# return the l1 value linked to the best performance
	bestl1 = l1s[performance.index(max(performance))]
	print("   Done! (Best L1: " + str(bestl1) + ")")
	return(bestl1)


# creates 10-fold cross validation set and then evaluates the performance
# over an increasing number of features 
def findOptimalFeatNmulti(trainfiles,cats,imageformat,featurefile,maxk,bestL1vals):
	# create results folders
	if not os.path.exists('results'):
		os.makedirs('results')

	if not os.path.exists('results/classification-crossvalidation'):
		os.makedirs('results/classification-crossvalidation')

	if not os.path.exists('results/classification-testset'):
		os.makedirs('results/classification-testset')

	if not os.path.exists('tmp'):
		os.makedirs('tmp')


	outfile = codecs.open('results/results-increasing-featurecount-crossvalidation.txt','w','utf-8')
	outfile2 = codecs.open('results/results-increasing-featurecount-testset.txt','w','utf-8')
	featureslist = []
	trainbases = []
	# generating the ranking using grafting on the separate dataset
	for i in range(0,len(trainfiles)):
		f = trainfiles[i]
		print(f)

		basename = f.replace('input/','').replace('.parsed.grafting','').replace('.txt','').replace('train','genre')
		
		# results in new input files, where features are excluded from the training data
		# if these are specified in excluded.txt 
		excludeFeatures(f,featurefile,'input/excluded.txt',basename)
		if bestL1vals == -1: 
			bestL1 = findBestL1(basename) # determine best L1
		else: 
			bestL1 = bestL1vals[i]

		trainbases.append(basename)

		# generate maxk-fold cross validation set, results in tmp/train-[basename]-#.txt and tmp/develop-[basename]-#.txt files
		generateKfold('tmp/training-'+basename+'.txt',basename,maxk)

		# training each of the final models, on the basis of the full dataset
		print("3. Training model using --l1 " + str(bestL1) + " --grafting 1 (be patient)...",end='')
		sys.stdout.flush() 
		outfile.write('Used L1: ' + str(bestL1) + '\n\n')
		train('tmp/training-'+basename+'.txt',bestL1,1,'tmp/weights-'+basename+'.txt','tmp/weights-'+basename+'.proc') # use grafting to obtain feature ranking
		print(" Done!")

		# sort features (1 feature per line and line numbers corresponding to ID's)
		print("4. Sorting features...", end='')
		features = sortedfeatnames(featurefile,'tmp/weights-'+basename+'.proc') 
		featureslist.append(features)
		sys.stdout.flush() 
	
	print("\n5. Training models for increasing feature set:")
	
	accuracies = []
	testaccs = []
	setops = []
	sedowns = []

	# determine maximum number of features
	maxfeatAllGenres = 0
	for i in range(0,len(trainbases)):
		basename = trainbases[i]

		# extract the first i features from the feature ranking (some features may have been culled during the l1 selection)
		maxfeatures = int(os.popen('grep \"^\*\" tmp/weights-'+basename+'.proc | wc -l').read().strip())
		maxfeatAllGenres = max(maxfeatAllGenres,maxfeatures)

	
	for i in range(1,maxfeatAllGenres+1):
		curlength = 0
		curacc = []
		fullweights = []
		for k in range(0,maxk):
			weightfiles = []
			curlength = 0
			for j in range(0,len(trainbases)):
				basename = trainbases[j]

				# extract the first i features from the feature ranking (some features may have been culled during the l1 selection)
				maxfeatures = int(os.popen('grep \"^\*\" tmp/weights-'+basename+'.proc | wc -l').read().strip())

				if i <= maxfeatures: # if this genre had more features culled and we have reached the maximum feature count, we don't redo training
					curlength += 1

					# extract top i features from the ranking
					os.system("grep \"^\*\" tmp/weights-"+basename+".proc | head -n " + str(i) + " | sed 's/^\* //g' > tmp/selected-features")

					# remove the unused features from the training set, they can be kept in the test set as the weights will be 0 from training when not used
					# and we need the values also for features not in the training set, as they may be in a training set for another genre
					os.system("./remove_fs.py tmp/selected-features < tmp/train-"+basename+"-"+str(k)+".txt" + " > tmp/train_subset.txt") 
					os.system("cp tmp/develop-"+basename+"-"+str(k)+".txt tmp/test_subset-" + basename + ".txt")
					os.system("perl -pi -e 's/^1 0.*\n//g' tmp/test_subset-" + basename + ".txt") # remove 1 0
					os.system("perl -pi -e 's/^2.*\n//g' tmp/test_subset-" + basename + ".txt") # remove 2
					os.system("perl -pi -e 's/^0.*\n//g' tmp/test_subset-" + basename + ".txt") # remove 0 .....
					os.system("perl -pi -e 's/^1 /" + cats[j]+" /g' tmp/test_subset-" + basename + ".txt") # replace 1 with category
					
					# train the model on the subset of features (without l1 and grafting) using the 10-fold crossvalidation training set 
					train('tmp/train_subset.txt',0,0,'tmp/weights-'+basename+'.txt','tmp/tmp.proc')

					if k == 0: # we also train the model on the full data set
						os.system("./remove_fs.py tmp/selected-features < tmp/training-"+basename+".txt" + " > tmp/train_subset_for_test.txt") 
						train('tmp/train_subset_for_test.txt',0,0,'tmp/weights-for-test-'+basename+'.txt','tmp/tmp-full-'+basename+'.proc')

				weightfiles.append('tmp/weights-'+basename+'.txt') # store weightfiles for the 4 genres, based on 10-fold cross validation set
				if k==0:
					fullweights.append('tmp/weights-for-test-'+basename+'.txt') 
			
			if curlength > 0: # if at least one additional feature added (should always be the case)
				# run test using the 10-fold cross validation test set, on the basis of weights obtained from the 10-fold cross validation training phase 
				os.system("cat tmp/test_subset-*.txt > tmp/test_subset.txt")
				curacc.append(multileveltest('tmp/test_subset.txt',weightfiles,cats,'results/classification-crossvalidation/classification-cv-'+ str(i) + '-' + str(k) +'.txt'))
		
			if k==0:
				# run test using the complete test set, on the basis of weights obtained from the full training set
				testacc = multileveltest('input/testing.grafting',fullweights,cats,'results/classification-testset/classification-test-'+ str(i) +'.txt')
 

		if curlength > 0:
			avgacc = m(curacc) # mean of 10-fold cross validation performance on 10-fold cv test set

			# create confidence interval
			if maxk > 1:
				setop = avgacc + 1.96*SE(curacc,avgacc)
				sedown = avgacc - 1.96*SE(curacc,avgacc)
			else:
				setop = avgacc
				sedown = avgacc
			
			accuracies.append(avgacc) # store accuracies associated with the number of features
			setops.append(setop)
			sedowns.append(sedown)
			testaccs.append(testacc) # similarly for accuracy on test set 

			if (i == 1):
				print(str(i) + " feature: " + str(avgacc)+ " (" + str(maxk) + "-fold) / " + str(testacc) + " (test)")
				print("\t[",end="")
				outfile.write(str(i) + " feature: " + str(avgacc) + " (" + str(maxk) + "-fold)\n\t[")
				outfile2.write(str(i) + " feature: " + str(testacc) + " (test)\n\t[")
				for j in range(0,len(featureslist)):
					if (j == (len(featureslist)-1)): 
						if (len(featureslist[j]) >= i) : 
							outfile.write(cats[j] + ' = ' + featureslist[j][i-1] + ']\n')
							outfile2.write(cats[j] + ' = ' + featureslist[j][i-1] + ']\n')
							print(cats[j] + ' = ' + unicodedata.normalize('NFKD', featureslist[j][i-1]).encode('ascii','ignore').decode('ascii') + ']\n',end="")
						else:
							outfile.write(cats[j] + ' = NONE]\n')
							outfile2.write(cats[j] + ' = NONE]\n')
							print(cats[j] + ' = NONE]\n',end="")
					else:
						if (len(featureslist[j]) >= i) : 
							outfile.write(cats[j] + ' = ' + featureslist[j][i-1] + ', ')
							outfile2.write(cats[j] + ' = ' + featureslist[j][i-1] + ', ')
							print(cats[j] + ' = ' + unicodedata.normalize('NFKD', featureslist[j][i-1]).encode('ascii','ignore').decode('ascii') + ', ',end="")
						else:
							outfile.write(cats[j] + ' = NONE, ')
							outfile2.write(cats[j] + ' = NONE, ')
							print(cats[j] + ' = NONE, ',end="")

			else:
				print(str(i) + " features: " + str(avgacc)+ " (" + str(maxk) + "-fold) / " + str(testacc) + " (test)")
				print("\t[+",end="")
				outfile.write(str(i) + " features: " + str(avgacc)+ " (" + str(maxk) + "-fold)\n\t[+ ")
				outfile2.write(str(i) + " features: " + str(testacc) + " (test)\n\t[+ ")
				for j in range(0,len(featureslist)):
					if (j == (len(featureslist)-1)): 
						if (len(featureslist[j]) >= i) : 
							outfile.write(cats[j] + ' = ' + featureslist[j][i-1] + ']\n')
							outfile2.write(cats[j] + ' = ' + featureslist[j][i-1] + ']\n')
							print(cats[j] + ' = ' + unicodedata.normalize('NFKD', featureslist[j][i-1]).encode('ascii','ignore').decode('ascii') + ']\n',end="")
						else:
							outfile.write(cats[j] + ' = NONE]\n')
							outfile2.write(cats[j] + ' = NONE]\n')
							print(cats[j] + ' = NONE]\n',end="")
					else:
						if (len(featureslist[j]) >= i) : 
							outfile.write(cats[j] + ' = ' + featureslist[j][i-1] + ', ')
							outfile2.write(cats[j] + ' = ' + featureslist[j][i-1] + ', ')
							print(cats[j] + ' = ' + unicodedata.normalize('NFKD', featureslist[j][i-1]).encode('ascii','ignore').decode('ascii') + ', ',end="")
						else:
							outfile.write(cats[j] + ' = NONE, ')
							outfile2.write(cats[j] + ' = NONE, ')
							print(cats[j] + ' = NONE, ',end="")
	
	# plot the performance on the cross validation test set and store in image file
	plt.figure(1)
	plt.plot(range(1,len(accuracies)+1),accuracies, 'r', label='10-fold')
	if (maxk > 1):
		plt.plot(range(1,len(accuracies)+1),setops, 'k--', label='_nolegend_')
		plt.plot(range(1,len(accuracies)+1),sedowns, 'k--', label='_nolegend_')
	plt.legend(loc=3)
	plt.ylim([0,100])
	plt.xlabel('Number of features')
	plt.ylabel('Accuracy (%)')
	plt.savefig('results/results-crossvalidation.'+imageformat, bbox_inches=0)

	plt.figure(2)
	plt.plot(range(1,len(testaccs)+1),testaccs, 'b', label='test')
	plt.legend(loc=3)
	plt.ylim([0,100])
	plt.xlabel('Number of features')
	plt.ylabel('Accuracy (%)')
	plt.savefig('results/results-testset.'+imageformat, bbox_inches=0)

	outfile.close()
	outfile2.close()
	print("\nCompleted!\n")

# Modifies trainingsfiles to exclude features which are specified
# to be excluded a priori. The numerical labels of the features
# are not changed, to not affect testing files
def excludeFeatures(trainfile, featurefile, exclusionfile, basename): 
	print("\n1. Excluding features from excluded.txt...",end='')
	
	features = []
	features.append("IGNORED")

	thefile = codecs.open(featurefile, 'r','utf-8')
	for line in thefile:
		features.append(line.strip())
	thefile.close()

	excluded = []
	exfile = open(exclusionfile, 'r')
	for line in exfile:
		excluded.append(line.strip())
	exfile.close()

	outfile = open('tmp/included.txt','w')
	for i in range(1,len(features)):
		if features[i] not in excluded:
			outfile.write(str(i) + "\n")
	outfile.close()

	# remove features which should be excluded a priori
	os.system("./remove_fs.py tmp/included.txt < " + trainfile + " > tmp/training-" + basename+".txt") 
	print(" Done!")


tinyestpath = ""
trainfiles = ['input/train_Journalism.parsed.grafting', 'input/train_Educational.parsed.grafting','input/train_ScientificProse.parsed.grafting','input/train_Literature.parsed.grafting'] # general training files
cats = ['JOU','EDU','SCI','LIT']
findOptimalFeatNmulti(trainfiles,cats,"png","input/FeatNames.txt",10,-1)
