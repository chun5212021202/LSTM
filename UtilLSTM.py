
import numpy as np
from random import shuffle
import json
from pprint import pprint

import time


#########################################################
#														#
#						FUNCTIONS						#
#														#
#########################################################

def LoadTrainLSTM(training_file_path) :
	#	read file
	print 'File \'json\' Loading.....'
	with open(training_file_path) as data_file:    
    		data = json.load(data_file)
	
	P = []
	Q = []
	A = []
	B = []
	C = []
	D = []
	Ans = []
	keyList = data.keys()
	for idx,article in enumerate(data['P']):
		P.append([])
		for word in article:
			word = word.split();
			word = map(float, word);	
			P[idx].append(word)

	for idx,question in enumerate(data['Q']):
		if idx%4 == 0:
			Q.append([])
		Q[idx/4].append([])
		for word in question:
			word = word.split()
			word = map(float, word)	
			Q[idx/4][idx%4].append(word)
	for idx,AnswerA in enumerate(data['A']):
		if idx%4 == 0:
			A.append([])
		A[idx/4].append([])
		for word in AnswerA:
			word = word.split();
			word = map(float, word);	
			A[idx/4][idx%4].append(word)
	for idx,AnswerB in enumerate(data['B']):
		if idx%4 == 0:
			B.append([])
		B[idx/4].append([])
		for word in AnswerB:
			word = word.split();
			word = map(float, word)	
			B[idx/4][idx%4].append(word)
	for idx,AnswerC in enumerate(data['C']):
		if idx%4 == 0:
			C.append([])
		C[idx/4].append([])
		for word in AnswerC:
			word = word.split()
			word = map(float, word)	
			C[idx/4][idx%4].append(word)
	for idx,AnswerD in enumerate(data['D']):
		if idx%4 == 0:
			D.append([])
		D[idx/4].append([])
		for word in AnswerD:
			word = word.split()
			word = map(float, word)	
			D[idx/4][idx%4].append(word)

	for idx,Answer in enumerate(data['Ans']):
		if idx%4 == 0:
			Ans.append([])
		Ans[idx/4].append([])
		for word in Answer:
			word = word.split()
			word = map(float, word)	
			Ans[idx/4][idx%4].append(word)



	"""			
	print len(P), len(Q), len(A), len(B), len(C), len(D), len(Ans)
	num,_ = np.asarray(Ans[1][2]).shape
	print np.asarray(P[1]+Q[1][2]).shape, np.asarray(Ans[1][2]).shape, (np.sum( np.asarray(Ans[1][2]),axis=0 )/num).shape, np.sum( np.asarray(Ans[1][2]),axis=1 ).shape, num
	print np.asarray(Ans[1][2]),np.sum( np.asarray(Ans[1][2]),axis=0 )/num
	"""
	return P,Q,A,B,C,D,Ans 




def TrainLSTM(data, lstm_object, epoch) :	# train one epoch

	P = data[0]
	Q = data[1]
	A = data[2]
	B = data[3]
	C = data[4]
	D = data[5]
	Ans = data[6]


	training_data_x = data[1]
	training_data_y = data[2]

	print ' *** START TRAINING *** '
	
	cost_count = []
	for turns in range(epoch) :
		cost_count.append(0)
		for i, paragraph in enumerate(P) :

			for j in range(4) :
				training_data_x = np.asarray(P[i]+Q[i][j])
				num,_ = np.asarray(Ans[i][j]).shape
				training_data_y = np.sum( np.asarray(Ans[i][j]), axis=0) / num

				start = time.time()
				temp = lstm_object.train( training_data_x, training_data_y )
				cost_count[turns]+=temp[0]
				print turns,'-',i,'-',j,' : ',temp[0],' len(Ans):',num, ">>> time:",time.time()-start
				print
	
	print 'total cost: ',cost_count


def TestLSTM(data, lstm_object) :	# train one epoch

	P = data[0]
	Q = data[1]
	A = data[2]
	B = data[3]
	C = data[4]
	D = data[5]
	Ans = data[6]


	training_data_x = data[1]
	training_data_y = data[2]

	print ' *** START TESTING *** '
	

	correct = 0
	for i, paragraph in enumerate(P) :

		for j in range(4) :
			start = time.time()

			testing_data_x = np.asarray(P[i]+Q[i][j])
			num,_ = np.asarray(Ans[i][j]).shape
			numA,_ = np.asarray(A[i][j]).shape
			numB,_ = np.asarray(B[i][j]).shape
			numC,_ = np.asarray(C[i][j]).shape
			numD,_ = np.asarray(D[i][j]).shape

			testing_answer = np.sum( np.asarray(Ans[i][j]), axis=0) / num
			testing_A = np.sum( np.asarray(A[i][j]), axis=0) / numA
			testing_B = np.sum( np.asarray(B[i][j]), axis=0) / numB
			testing_C = np.sum( np.asarray(C[i][j]), axis=0) / numC
			testing_D = np.sum( np.asarray(D[i][j]), axis=0) / numD
			if np.array_equal(testing_answer, testing_A) :
				answer = 0
			elif np.array_equal(testing_answer, testing_B) :
				answer = 1
			elif np.array_equal(testing_answer, testing_C) :
				answer = 2
			elif np.array_equal(testing_answer, testing_D) :
				answer = 3
			
			temp = lstm_object.test( testing_data_x )

			consistency_A = np.sum(np.absolute(temp-testing_A))
			consistency_B = np.sum(np.absolute(temp-testing_B))
			consistency_C = np.sum(np.absolute(temp-testing_C))
			consistency_D = np.sum(np.absolute(temp-testing_D))

			n = [consistency_A, consistency_B, consistency_C, consistency_D]
			predict = n.index(min(n))
			if predict == answer :
				correct+=1

			print i,'-',j,' : answer>',answer,' predict>',predict, ">>> time:",time.time()-start
			print
	
	print 'Total Correct: ', correct
	print 'Total Q: ', len(data[1])
