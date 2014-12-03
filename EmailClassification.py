import json
from collections import defaultdict
import nltk
import numpy as np
import string as string
import stemming
from nltk.corpus import stopwords
import nltk.metrics
import glob
import os
import copy
from stemming.porter2 import stem
import random
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from math import log
import timeit



def string_mani(lst, att):
	cachedStopWords = stopwords.words("english")
	cahcedStopWords = [ str(j) for j in cachedStopWords]
	j = 0
	full_term_list = []
	orig_lst = lst

	for i in range(len(lst)):
		if type(lst[i]) == list or type(lst[i]) == dict:
			if len(lst[i]) < 1:
				lst[i] = unicode("null")
			elif len(lst[i]) > 1:
				lst[i] = set(lst[i])
		else:
			lst = lst
		encoded_string = lst[i]
		encoded_string = unicode(encoded_string)
		encoded_string = encoded_string.encode('utf-8')
		# type(encoded_string) becomes str
		encoded_string = encoded_string.lower()

		encoded_string = encoded_string.replace("set"," ")
		encoded_string = encoded_string.replace("bounce","bounce ")
		encoded_string = encoded_string.replace("no-reply","noreply ")
		encoded_string = encoded_string.replace("do_not_reply", "donotreply")
		#replacing punctuations with spaces
		replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
		encoded_string = encoded_string.translate(replace_punctuation)

		#removing numbers
		encoded_string = ''.join(i for i in encoded_string if not i.isdigit())
		#stemming
		encoded_string = [stem(word) for word in encoded_string.split()]
		#removing stop words
		# caching the stopwords object
		encoded_string = [word for word in encoded_string if word not in cachedStopWords]
		encoded_string = [encoded_string[i] for i in range(len(encoded_string)) if len(encoded_string[i])>1]

		if att == 'return-path':
			encoded_string = ['return_path.' + word for word in encoded_string]
			full_term_list.append(encoded_string[:])
		elif att == 'subject':
			encoded_string = ['subject.' + word for word in encoded_string]
			full_term_list.append(encoded_string[:])
		elif att == 'from':
			encoded_string = ['from.' + word for word in encoded_string]
			full_term_list.append(encoded_string[:])
	return full_term_list, orig_lst

# Andrea's JSON format: Labels directly under [mail] and [headers]
# Returns the following items for each individual email: [0]label_lst, [2]return_list, [4]subject_list
# Returns the following items for entire cohort: [1]all label_terms, [3]return_terms [5] subject_terms, [6]total_counts
def tdm(dataset):

	label = []
	label_lst = []
	subject_lst =[]
	return_lst =[]
	from_lst = []

	for i in range(len(dataset)):
		# label_lst2 = []
		# label_lst = []

		# for k, v in dataset[i]['mail']['headers'].items():
		# 	if v !=[None] and v !={}:
		# 		label_lst2.append(k)

		test = dataset[i]['mail']['headers']
		test.keys()
		label_lst.append(test.keys())
		subject_lst.append(test["subject"])
		from_lst.append(test["from"])

		if 'return-path' not in test.keys():
			return_lst.append(unicode("null"))
		else:
			return_lst.append(test["return-path"])

	for i in range(len(label_lst)):
		for j in range(len(label_lst[i])):
				label_lst[i][j] = 'headers.'+ label_lst[i][j]
				label_lst[i][j] = label_lst[i][j].encode('utf-8')

	return_list = string_mani(return_lst, 'return-path')
	subject_list = string_mani(subject_lst, 'subject')
	from_list = string_mani(from_lst, 'from')

	# count occurrences for all terms	
	all_label = [str(item) for sublist in label_lst for item in sublist]
	all_return = [item for sublist in return_list[0] for item in sublist]
	all_subject = [item for sublist in subject_list[0] for item in sublist]
	all_from = [item for sublist in from_list[0] for item in sublist]
	

	# append count for header labels to total_counts dict
	total_counts = {}

	for t in set(all_label):
		total_counts[t] = all_label.count(t)
	# append count for all return terms to total_counts dict
	for t in set(all_return):
		total_counts[t]= all_return.count(t)

	# append count for all subject terms to total_counts dict
	for t in set(all_subject):
		total_counts[t]= all_subject.count(t)

	for t in set(all_from):
		total_counts[t]= all_from.count(t)	

	# remove special features
 	total_counts_trim_s = {k:v for k, v in total_counts.items() if 'stit' not in k and 'somrat' not in k and 'niyogi' not in k and 'sanjay' not in k and\
 							'wiebk' not in k and 'poerschk' not in k and 'hothouselab' not in k and 'ivan' not in k and 'katrina' not in k and 'wong' not in k and\
 							'greg' not in k and 'robinson' not in k and 'jason' not in k and 'mcdowal' not in k and 'maria' not in k and 'andrea' not in k and 'sandberg' not in k\
 							and 'allen' not in k and 'yuklai' not in k and 'suen' not in k and 'mei' not in k and\
 							 'stephen' not in k and k != 'kim' and 'artur' not in k and 'kesel' not in k and 'andr' not in k and 'green' not in k and\
 							 'davidparrish' not in k and 'david' not in k and 'parrish' not in k}

	# remove return-path and subject features with frequency count less than 3 (411 features remaining)

	total_counts_trim_f = {}
	for key in total_counts_trim_s:
		if total_counts_trim_s[key] > 2 :
			total_counts_trim_f[key]= total_counts_trim_s[key]

	t = []
	for i in all_return:
		if i not in t:
			t.append(i)

	s = []
	for i in all_subject:
		if i not in s:
			s.append(i)

	l = []
	for i in all_label:
		if i not in l:
			l.append(i)
	f = []
	for i in all_from:
		if i not in f:
			f.append(i)

	label_terms = [i for i in l if i in total_counts_trim_f]
	return_terms = [i for i in t if i in total_counts_trim_f]
	subject_terms = [i for i in s if i in total_counts_trim_f]
	from_terms = [i for i in f if i in total_counts_trim_f]	

	return label_lst, label_terms, return_list[0], return_terms, subject_list[0], subject_terms,\
	 from_list[0], from_terms, all_from, total_counts_trim_f, total_counts
	# , total_counts





# Clean Data JSON Format
def tdm2(dataset):


	label_lst = []
	subject_lst =[]
	return_lst =[]
	from_lst = []
	# exclude_list_y = ['headers.x-received.y','headers.delivered-to.y','headers.domainkey-signature.y',\
	# 'headers.x-beenthere.y','headers.received-spf.y','headers.received.y','headers.dkim-signature.y','headers.return-path.y']
	# exclude_list_x =['headers.x-received.x','headers.delivered-to.x','headers.domainkey-signature.x',\
	# 'headers.x-beenthere.x','headers.received-spf.x','headers.received.x',\
	# 'headers.dkim-signature.x','headers.return-path.x']


	for i in range(len(dataset)):
		hxy =[]
		for k, v in dataset[i].items():
			if 'headers.' in k and v !=[None] and v !={}:
				hxy.append(k)
				# print k, '::', v

		h = []
		for m in range(len(hxy)):
			if hxy[m] == ('headers.x-received.x') or hxy[m] ==('headers.x-received.y'):
				if 'headers.x-received' not in h:
					h.append(unicode('headers.x-received'))
			elif hxy[m] == ('headers.delivered-to.x') or hxy[m] ==('headers.delivered-to.y'):
				if 'headers.delivered-to' not in h:
					h.append(unicode('headers.delivered-to'))
			elif hxy[m] == ('headers.domainkey-signature.x') or hxy[m] ==('headers.domainkey-signature.y'):
				if 'headers.domainkey-signature' not in h:
					h.append(unicode('headers.domainkey-signature'))
			elif hxy[m] == ('headers.x-beenthere.x') or hxy[m] ==('headers.x-beenthere.y'):
				if 'headers.x-beenthere' not in h:
					h.append(unicode('headers.x-beenthere'))
			elif hxy[m] == ('headers.received-spf.x') or hxy[m] ==('headers.received-spf.y'):
				if 'headers.received-spf' not in h:
					h.append(unicode('headers.received-spf'))
			elif hxy[m] == ('headers.received.x') or hxy[m] ==('headers.received.y'):
				if 'headers.received' not in h:
					h.append(unicode('headers.received'))
			elif hxy[m] == ('headers.dkim-signature.x') or hxy[m] ==('headers.dkim-signature.y'):
				if 'headers.dkim-signature' not in h:
					h.append(unicode('headers.dkim-signature'))
			elif hxy[m] == ('headers.return-path.x') or hxy[m] ==('headers.return-path.y'):
				if 'headers.return-path' not in h:
					h.append(unicode('headers.return-path'))
			else:
				h.append(hxy[m])

		h = [h[i].encode('utf-8') for i in range(len(h))]
		label_lst.append(h)
		subject_lst.append(dataset[i]['headers.subject'])
		from_lst.append(dataset[i]['headers.from'])

		if 'headers.return-path' not in dataset[i].keys():
			x = dataset[i]['headers.return-path.x']
			y = dataset[i]['headers.return-path.y']
			if x == {} and y != {}:
				return_lst.append(y)
			elif y == {} and x != {}:
				return_lst.append(x)
			else:
				return_lst.append(y + x)
		else:
			return_lst.append(dataset[i]['headers.return-path'])
		
	# label_terms = ([str(item) for sublist in label_lst for item in sublist])
	return_list = string_mani(return_lst, 'return-path')
	subject_list = string_mani(subject_lst, 'subject')
	from_list = string_mani(from_lst, 'from')
	
	# count occurrences for all terms
	all_label = [str(item) for sublist in label_lst for item in sublist]
	all_return = [item for sublist in return_list[0] for item in sublist]
	all_subject = [item for sublist in subject_list[0] for item in sublist]
	all_from = [item for sublist in from_list[0] for item in sublist]

	# append count for header labels to total_counts dict
	total_counts = {}

	for t in set(all_label):
		total_counts[t] = all_label.count(t)
	# append count for all return terms to total_counts dict
	for t in set(all_return):
		total_counts[t]= all_return.count(t)

	# append count for all subject terms to total_counts dict
	for t in set(all_subject):
		total_counts[t]= all_subject.count(t)

	for t in set(all_from):
		total_counts[t]= all_from.count(t)

	# remove special features
 	total_counts_trim_s = {k:v for k, v in total_counts.items() if 'stit' not in k and 'somrat' not in k and 'niyogi' not in k and 'sanjay' not in k and\
 							'wiebk' not in k and 'poerschk' not in k and 'hothouselab' not in k and 'ivan' not in k and 'katrina' not in k and 'wong' not in k and\
 							'greg' not in k and 'robinson' not in k and 'jason' not in k and 'mcdowal' not in k and 'maria' not in k and 'andrea' not in k and 'sandberg' not in k\
 							and 'allen' not in k and 'yuklai' not in k and 'suen' not in k and 'mei' not in k and\
 							 'stephen' not in k and k != 'kim' and 'artur' not in k and 'kesel' not in k and 'andr' not in k and 'green' not in k and\
 							 'davidparrish' not in k and 'david' not in k and 'parrish' not in k}

	# remove return-path and subject features with frequency count less than 3 (411 features remaining)

	total_counts_trim_f = {}
	for key in total_counts_trim_s:
		if total_counts_trim_s[key] > 2 :
			total_counts_trim_f[key]= total_counts_trim_s[key]

	# keep all header label features
	# for t in set(all_label):
	# 	total_counts_trim[t] = all_label.count(t)


	t = []
	for i in all_return:
		if i not in t:
			t.append(i)

	s = []
	for i in all_subject:
		if i not in s:
			s.append(i)

	l = []
	for i in all_label:
		if i not in l:
			l.append(i)

	f = []
	for i in all_from:
		if i not in f:
			f.append(i)

	label_terms = [i for i in l if i in total_counts_trim_f]
	return_terms = [i for i in t if i in total_counts_trim_f]
	subject_terms = [i for i in s if i in total_counts_trim_f]
	from_terms = [i for i in f if i in total_counts_trim_f]	

	return label_lst, label_terms, return_list[0], return_terms, subject_list[0], subject_terms, from_list[0], from_terms, total_counts_trim_f, total_counts



def slicedict(sub,h):
	return {k:v for k,v in sub.iteritems() if k.startswith(h)}

def ClassLabel(dataset):
	c = []
	for i in range(len(dataset)):
		# print dataset[i][1]
		c.append(dataset[i][1])
	print '# of human instances: %d, As percentage = %.2f' %(c.count('human'), round(float(c.count('human'))/float(len(dataset))*100, 4)),'%'
	print '# of auto instances: %d, As percentage = %.2f' %(c.count('auto'), round(float(c.count('auto'))/float(len(dataset))*100, 4)),'%'
	h = float(c.count('human'))/float(len(dataset))
	a = float(c.count('auto'))/float(len(dataset))
	return h, a

def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
        cpdist = self._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l,fname].prob(fval)

            labels = sorted([l for l in self._labels
                             if fval in cpdist[l,fname].samples()],
                            key=labelprob)
            if len(labels) == 1: continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0,fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /
                                  cpdist[l0,fname].prob(fval))
            print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))


def most_informative_features(self, n=100):
	"""
	Return a list of the 'most informative' features used by this
	classifier.  For the purpose of this function, the
	informativeness of a feature ``(fname,fval)`` is equal to the
	highest value of P(fname=fval|label), for any label, divided by
	the lowest value of P(fname=fval|label), for any label:

	|  max[ P(fname=fval|label1) / P(fname=fval|label2) ]
	"""
	# The set of (fname, fval) pairs used by this classifier.
	features = set()
	# The max & min probability associated w/ each (fname, fval)
	# pair.  Maps (fname,fval) -> float.
	maxprob = defaultdict(lambda: 0.0)
	minprob = defaultdict(lambda: 1.0)

	for (label, fname), probdist in self._feature_probdist.items():
		for fval in probdist.samples():
			feature = (fname, fval)
			features.add( feature )
			p = probdist.prob(fval)
			print p
			maxprob[feature] = max(p, maxprob[feature])
			minprob[feature] = min(p, minprob[feature])
			if minprob[feature] == 0:
				features.discard(feature)
			# print maxprob
			# print minprob


	# Convert features to a list, & sort it by how informative
	# features are.
	features = sorted(features,
	    key=lambda feature_: minprob[feature_]/maxprob[feature_])
	return features[:n]


def email_pdist(dataset,j):
	term_lst = []
	cpdist = classifierNB._feature_probdist
	cpdist_terms = [cpdist.items()[t][0][1] for t in range(len(cpdist))]
	fval = 1
	auto_prob =1.0
	human_prob = 1.0
	for i in range(0,7,2):
		term_lst.append(dataset[i][j])
	term_lst = [ i for sublst in term_lst for i in sublst]
	print term_lst
	for fname in term_lst:
			if fname in cpdist_terms:
				print 'human_prob| %s : %f' %(fname, cpdist['human',fname].prob(fval))
				human_prob *= cpdist['human',fname].prob(fval)
				print 'auto_prob| %s : %f' %(fname, cpdist['auto',fname].prob(fval))
				auto_prob *= cpdist['auto',fname].prob(fval)
	if human_prob > auto_prob:
		print 'email is human'
	elif human_prob < auto_prob:
		print 'email is auto'
	else:
		print 'Undetermined'
	print 'prob of human= ',human_prob
	print 'prob of auto= ', auto_prob
	# print ClassLabel(feat_matrix)[0]* human_prob, ClassLabel(feat_matrix)[1]*auto_prob

	
"""
===========================================================================================================
main
===========================================================================================================
"""

if __name__ == '__main__':

	print "Running Naive Bayes Classification On Emails..."

	"""
	Reading the data in from files.
	"""

	auto_path = '/Users/heymanhn/CleanAutocopy.JSON'
	human_path = '/Users/heymanhn/CleanHumancopy.JSON'

	with open(auto_path) as b:
		email_full_auto = json.load(b)

	with open(human_path) as g:
		email_full_human = json.load(g)

	# instances without subject line. 1286 to 1282
	email_full_human.pop(1021)
	email_full_human.pop(1082)
	email_full_human.pop(1082)


	print 'Number of Machine Emails: ',len(email_full_auto)
	print 'Number of Human Emails: ', len(email_full_human)

	"""
	Building feature matrix for training set

	"""
	# Run each dataset through tdm function to get feature sets

	print "Performance of running tdm2 on human training set: ", timeit.timeit("tdm2(email_full_human)", setup = "from __main__ import tdm2, email_full_human", number=1)
	print "Performance of running tdm2 on auto training set: ", timeit.timeit("tdm2(email_full_auto)", setup = "from __main__ import tdm2, email_full_auto", number=1)
	
	auto = tdm2(email_full_auto)
	human = tdm2(email_full_human)

	total_label_term = []
	# combining label features from human and auto
	# 157 features
	for i in human[1]:
		if i not in total_label_term:
			total_label_term.append(i)
	for i in auto[1]:
		if i not in total_label_term:
			total_label_term.append(i)

	total_return_term = []
	# combining return-path features from human and auto
	# 134 features
	for i in human[3]:
		if i not in total_return_term :
			total_return_term.append(i)
	for i in auto[3]:
		if i not in total_return_term :
			total_return_term.append(i)

	total_subject_term = []
	# combining subject features from human and auto
	# 
	for i in human[5]:
		if i not in total_subject_term:
			total_subject_term.append(i)

	for i in auto[5]:
		if i not in total_subject_term:
			total_subject_term.append(i)

	total_from_term = []
	# combining subject features from human and auto
	# 446
	for i in human[7]:
		if i not in total_from_term:
			total_from_term.append(i)

	for i in auto[7]:
		if i not in total_from_term:
			total_from_term.append(i)

	# feat_dict consists of feature binary indicator for all instances( 1 for True, 0 for False)
	# total of 737 features
	# feat_matrix is a tuple consists of (feat_dict,label). feat_matrix[0:1281] are human, feat_matrix[282:2005] are auto
	feat_matrix = []
	feat_dict = {}

	for n in range(len(email_full_human)): 
		for m, label_term in enumerate(total_label_term):
			feat_dict[label_term] = human[0][n].count(label_term)
		for m, return_term in enumerate(total_return_term):
			feat_dict[return_term] = human[2][n].count(return_term)
		for m, subject_term in enumerate(total_subject_term):
			feat_dict[subject_term] = human[4][n].count(subject_term)
		for m, from_term in enumerate(total_from_term):
			feat_dict[from_term] = human[6][n].count(from_term)
			feat_tuple = (feat_dict.copy(),'human')
		feat_matrix.append(feat_tuple)

	for n in range(len(email_full_auto)): 
		for m, label_term in enumerate(total_label_term):
			feat_dict[label_term] = auto[0][n].count(label_term)
		for m, return_term in enumerate(total_return_term):
			feat_dict[return_term] = auto[2][n].count(return_term)
		for m, subject_term in enumerate(total_subject_term):
			feat_dict[subject_term] = auto[4][n].count(subject_term)
		for m, from_term in enumerate(total_from_term):
			feat_dict[from_term] = auto[6][n].count(from_term)
			feat_tuple = (feat_dict.copy(),'auto')
		feat_matrix.append(feat_tuple)

	# feat_matrix_string = str(feat_matrix)

	# with open("feat_matrix.txt", "w") as feat_matrix_file:
	# 		feat_matrix_file.write(feat_matrix_string)

	# full_set consists of shuffled version of feat_matrix
	full_set = feat_matrix
	random.shuffle(full_set)

	train_set = full_set[:]

	"""
	Importing test set.
	Andreas data set
	"""

	with open('bad.json') as b:
		auto_email_test = json.load(b)

	with open('good.json') as g:
		human_email_test= json.load(g)

	email_test = human_email_test + auto_email_test

	print 'Number of emails to be classified: ', len(email_test)

	"""
	Building feature matrix for Test Set. 
	"""

	test_set = tdm(email_test)

	test_feat_dict={}
	test_feat_matrix= []

	for n in range(len(email_test)): 
		for m, label_term in enumerate(total_label_term):
			test_feat_dict[label_term] = test_set[0][n].count(label_term)
		for m, return_term in enumerate(total_return_term):
			test_feat_dict[return_term] = test_set[2][n].count(return_term)
		for m, subject_term in enumerate(total_subject_term):
			test_feat_dict[subject_term] = test_set[4][n].count(subject_term)
		for m, from_term in enumerate(total_from_term):
			test_feat_dict[from_term] = test_set[6][n].count(from_term)
		test_feat_matrix.append(test_feat_dict.copy())


	print 'Label Distribution in Training Set'
	ClassLabel(train_set)
	print '============================='

	classifierNB = nltk.NaiveBayesClassifier.train(train_set)
	# features only data set. Excluded Label
	# test= [test_set[i][0] for i in range(len(test_set))]
	test = test_feat_matrix


	print 'Naive Bayes Classifier:'

	for pdist in classifierNB.prob_classify_many(test):
		print pdist.prob('human'), pdist.prob('auto')

	for i in range(len(classifierNB.classify_many(test))):
		print classifierNB.classify_many(test)[i]

	# Testing Accuracy
	# nb = nltk.classify.accuracy(classifierNB, test_set)
	# print 'accuracy is %.2f' %round(nb*100,4), '%'
	def NB():
		classifierNB = nltk.NaiveBayesClassifier.train(train_set)
		return classifierNB.classify_many(test)


	print "Performance of running Naive Bayes Classifier on test set: ", timeit.timeit("NB", setup = "from __main__ import NB", number = 1)




	"""
	Linear (Bernoulli) SVC
	Implementation of Support Vector Machine classifier using libsvm: 
	the kernel can be non-linear but its SMO algorithm does not scale to
	 large number of samples as LinearSVC does.
	"""

	 	
	from nltk.classify import SklearnClassifier
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.svm import SVC

	print ' '
	print '============================='
	print 'Bernoulli SVC Classifier:'
	classifierBi = SklearnClassifier(BernoulliNB()).train(train_set)
	classifierBi.classify_many(test)


	for pdist in classifierBi.prob_classify_many(test):
		print pdist.prob('human'), pdist.prob('auto')

	for i in range(len(classifierBi.classify_many(test))):
		print classifierBi.classify_many(test)[i]

	classifierSVC = SklearnClassifier(SVC(), sparse=True).train(train_set)
	classifierSVC.classify_many(test)

	# svc = nltk.classify.accuracy(classifierSVC, test_set)
	# print 'accuracy is %.2f' %round(svc*100,4), '%'
	def SVC():
		classifierBi = SklearnClassifier(BernoulliNB()).train(train_set)
		return classifierSVC.classify_many(test)


	print "Performance of running Bernoulli SVC Classifier on test set: ", timeit.timeit("SVC", setup = "from __main__ import SVC", number = 1)		

	print ' '
	print '============================='
	print 'Linear SVC Classifier:'		
	classifierLinSVC = SklearnClassifier(LinearSVC(), sparse=False).train(train_set)
	classifierLinSVC.classify_many(test)


	for i in range(len(classifierLinSVC.classify_many(test))):
		print classifierLinSVC.classify_many(test)[i]

	def LinSVC():
		classifierLinSVC = SklearnClassifier(LinearSVC(), sparse=False).train(train_set)
		return classifierLinSVC.classify_many(test)


	print "Performance of running Linear SVC Classifier on test set: ", timeit.timeit("LinSVC", setup = "from __main__ import LinSVC", number = 1)		


	# lin_svc = nltk.classify.accuracy(classifierLinSVC, test_set)
	# print 'accuracy is %.2f' %round(lin_svc*100,4), '%'	
	'''
	wrapping NB classifier with tf-idf weighing and chi-square features selection 
	'''
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.feature_selection import SelectKBest, chi2
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.pipeline import Pipeline
	pipeline = Pipeline([('tfidf', TfidfTransformer()), ('chi2', SelectKBest(chi2, k='all')),('nb', MultinomialNB())])
	classifierTF = SklearnClassifier(pipeline).train(train_set)

	print ' '
	print '============='
	print 'Naive Bayes with tf-idf weighing Classifier:'
	classifierTF.classify_many(test)


	for pdist in classifierTF.prob_classify_many(test):
		print pdist.prob('human'), pdist.prob('auto')

	for i in range(len(classifierTF.classify_many(test))):
		print classifierTF.classify_many(test)[i]

	# tf_class = nltk.classify.accuracy(classifierTF, test_set)
	# print 'accuracy is %.2f' %round(tf_class*100,4), '%'
	def NBtfidf():
		classifierTF = SklearnClassifier(pipeline).train(train_set)
		return classifierTF.classify_many(test)


	print "Performance of running Naive Bayes TF-IDF Classifier on test set: ", timeit.timeit("NBtfidf", setup = "from __main__ import NBtfidf", number = 1)		


	# Individual Email Feature Prob Dist.
	# email_pdist(test_set,0)





