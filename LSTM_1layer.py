import theano
import theano.tensor as T
import numpy as np
from itertools import izip

class LSTM (object ):
	def __init__ (self, input_dimension, output_dimension):	# input_dimension should be 2*x_dimension

		self.ParameterInit(input_dimension, output_dimension)




	def PlusNode(self, Para1, Para2):
		return Para1 + Para2

	def MultiplyNode(self, Para1, Para2):
		return Para1 * Para2

	def Sigmoid(self, z):
		return 1 / (1 + T.exp(-z))

	def Tanh(self, z):
		return T.tanh(z)

	def ParameterInit(self, input_dimension, output_dimension):
		self.W = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.B = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wf = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bf = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wi = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bi = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))
		self.Wo = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension, output_dimension+input_dimension)),dtype = theano.config.floatX))
		self.Bo = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(output_dimension)),dtype = theano.config.floatX))


	def InputGate(self, Control, Flow):		#flow: x, h_t-1
		_i = self.Sigmoid( T.dot(self.Wi, Control) + self.Bi )
		_input = self.Tanh( T.dot(self.W, Flow) + self.B )
		return self.MultiplyNode(_i, _input)

	def OutputGate(self, Control, Flow):	#flow: memory
		_o = self.Sigmoid( T.dot(self.Wo, Control) + self.Bo )
		_input = self.Tanh(Flow)
		return self.MultiplyNode(_o, _input)

	def ForgetGate(self, Control, Flow):
		_f = self.Sigmoid( T.dot(self.Wf, Control) + self.Bf )
		_input = Flow
		return self.MultiplyNode(_f, _input)


	def Get_C_and_H(self, x_t, h_tm1, c_tm1):
		signal = T.concatenate([h_tm1, x_t],axis=0)
		c_t = self.PlusNode( self.InputGate(signal, signal), self.ForgetGate(signal, c_tm1) )
		h_t = self.OutputGate( signal, c_t )
		return h_t, c_t
			





class NeuralNetwork (object):
	def __init__ (self, depth, h_dimension, x_dimension, y_dimension, learning_rate, alpha, grad_bound):
		self.LR = learning_rate
		self.ALPHA = alpha


		x_seq = T.matrix("x")
		y_hat_seq = T.vector("y")

		self.Network = LSTM(x_dimension, y_dimension)




		def step(x_t, h_out_tm1, c_out_tm1):

			#h_output = np.array( np.zeros((y_dimension )), dtype=theano.config.floatX)
			#c_output = np.array( np.zeros((y_dimension )), dtype=theano.config.floatX)

			h_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[0]
			c_output = self.Network.Get_C_and_H(x_t, h_out_tm1, c_out_tm1)[1]


			return h_output, c_output


				
		c_out_init = theano.shared(np.array( np.zeros((y_dimension)), dtype = theano.config.floatX))
		h_out_init = T.zeros_like(c_out_init)#theano.shared(np.array( np.zeros((y_dimension)), dtype = theano.config.floatX))


		[h_output_seq, c_output_seq],_ = theano.scan(	# frame number, depth, h_dimension
			step,
			sequences = x_seq,
			outputs_info = [h_out_init, c_out_init]
			)

		y_seq = self.softmax(h_output_seq)	# y_seq(frame, y_dim)

		cost = T.sum(np.absolute(h_output_seq[-1] - y_hat_seq))

		#dW,dB,dWf,dBf,dWi,dBi,dWo,dBo = [],[],[],[],[],[],[],[]

		dW= T.grad(cost, self.Network.W) 
		dB= T.grad(cost, self.Network.B) 
		dWf= T.grad(cost, self.Network.Wf) 
		dBf= T.grad(cost, self.Network.Bf) 
		dWi= T.grad(cost, self.Network.Wi) 
		dBi= T.grad(cost, self.Network.Bi) 
		dWo= T.grad(cost, self.Network.Wo) 
		dBo= T.grad(cost, self.Network.Bo) 



		parameters = []
		gradients = []

		parameters.append(self.Network.W)
		parameters.append(self.Network.B)
		parameters.append(self.Network.Wi)
		parameters.append(self.Network.Bi)
		parameters.append(self.Network.Wo)
		parameters.append(self.Network.Bo)
		parameters.append(self.Network.Wf)
		parameters.append(self.Network.Bf)
		gradients.append(dW)
		gradients.append(dB)
		gradients.append(dWi)
		gradients.append(dBi)
		gradients.append(dWo)
		gradients.append(dBo)
		gradients.append(dWf)
		gradients.append(dBf)
		


		self.update_parameter = theano.function(
			inputs = [x_seq,y_hat_seq],
			updates = self.UpdateParameter_RMSprop(parameters, gradients),
			outputs = [cost],
			allow_input_downcast = True
			)


		self.predict = theano.function(
			inputs = [x_seq],
			outputs = h_output_seq[-1],
			allow_input_downcast = True
			)




	def softmax(self, h):
		total = T.sum(T.exp(h),axis=1)
		return T.exp(h)/total.dimshuffle(0,'x')

	def UpdateParameter_RMSprop(self, parameter, gradient) :
		update = []
		for p,g in izip(parameter, gradient) :
			acc = theano.shared(p.get_value() * 0.)		# called once
			acc_new = self.ALPHA * acc + (1 - self.ALPHA) * g ** 2		# called once
        	
			scale = T.sqrt(acc_new + 1e-6)
			g = g/scale

			update += [(acc, acc_new)]
			update += [(p, p - self.LR*g)]

		return update

	def train (self, training_x_seq, training_y_seq):	# training_x_seq (frame, x_dim)		 training_y_seq (frame, y_dim)
		return self.update_parameter(training_x_seq, training_y_seq)



	def test (self, testing_x_seq):
		return self.predict(testing_x_seq)