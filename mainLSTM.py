import theano
import numpy as np
import theano.tensor as T
import UtilLSTM
import LSTM_3layer
import time

start = time.time()

LSTM_EPOCH = 20
LSTM_DEPTH = 3#2

LSTM_H_DIMENSION = 256	# width
LSTM_X_DIMENSION = 100
LSTM_Y_DIMENSION = 100
LSTM_LEARNING_RATE = 0.001


LSTM_DECAY = 0.99994#0.9999987
LSTM_ALPHA = 0.99
LSTM_GRAD_BOUND = 0.1
LSTM_OUTPUT_FILE = 'ResultLSTM/result_test.lab'



lstm = LSTM_3layer.NeuralNetwork(
	LSTM_DEPTH, 
	LSTM_H_DIMENSION, 
	LSTM_X_DIMENSION, 
	LSTM_Y_DIMENSION, 
	LSTM_LEARNING_RATE, 
	LSTM_ALPHA,
	LSTM_GRAD_BOUND
	)
print 'version: LSTM_3layer'
print 'NeuralNetwork Construction Time >>>>>',time.time() - start, 'sec'

data = UtilLSTM.LoadTrainLSTM('mc160.train.json')
UtilLSTM.TrainLSTM( data , lstm, LSTM_EPOCH )
UtilLSTM.TestLSTM( data , lstm )

data2 = UtilLSTM.LoadTrainLSTM('mc160.dev.json')
UtilLSTM.TestLSTM( data2 , lstm )
