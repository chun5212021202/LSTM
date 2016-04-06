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

		self.Network = []
		for i in range(depth):
			if i == depth-1 :
				self.Network.append( LSTM(h_dimension, y_dimension) ) 		# both 'hidden' and 'input' are both x_dimension
			elif i == 0 :
				self.Network.append( LSTM(x_dimension, h_dimension) )
			else :
				self.Network.append( LSTM(h_dimension, h_dimension) ) 



		def step(x_t, h_in_tm1, c_in_tm1, h_tm1, c_tm1, h_out_tm1, c_out_tm1):

			h_input = np.array( np.zeros((h_dimension )), dtype=theano.config.floatX)
			c_input = np.array( np.zeros((h_dimension )), dtype=theano.config.floatX)
			h_layer = np.array( np.zeros((h_dimension )), dtype=theano.config.floatX)
			c_layer = np.array( np.zeros((h_dimension )), dtype=theano.config.floatX)
			h_output = np.array( np.zeros((y_dimension )), dtype=theano.config.floatX)
			c_output = np.array( np.zeros((y_dimension )), dtype=theano.config.floatX)

			for i in range(depth):
					# connect all the LSTM
				if i == 0 :
					h_input = self.Network[i].Get_C_and_H(x_t, h_in_tm1, c_in_tm1)[0]  	# h_t
					c_input = self.Network[i].Get_C_and_H(x_t, h_in_tm1, c_in_tm1)[1]	# c_t
				elif i == depth-1 :
					h_output = self.Network[i].Get_C_and_H(h_layer, h_out_tm1, c_out_tm1)[0]
					c_output = self.Network[i].Get_C_and_H(h_layer, h_out_tm1, c_out_tm1)[1]
				else :
					h_layer = self.Network[i].Get_C_and_H(h_input, h_tm1, c_tm1)[0]	# h_t
					c_layer = self.Network[i].Get_C_and_H(h_input, h_tm1, c_tm1)[1]	# c_t

			return h_input, c_input ,h_layer, c_layer, h_output, c_output


		
		c_in_init = theano.shared(np.array( np.zeros((h_dimension)), dtype = theano.config.floatX))
		h_in_init = T.zeros_like(c_in_init)#theano.shared(np.array( np.zeros((h_dimension)), dtype = theano.config.floatX))
		
		c_init = theano.shared(np.array( np.zeros((h_dimension)), dtype = theano.config.floatX))
		h_init = T.zeros_like(c_init)#theano.shared(np.array( np.zeros((h_dimension)), dtype = theano.config.floatX))
		
		c_out_init = theano.shared(np.array( np.zeros((y_dimension)), dtype = theano.config.floatX))
		h_out_init = T.zeros_like(c_out_init)#theano.shared(np.array( np.zeros((y_dimension)), dtype = theano.config.floatX))


		[h_input_seq, c_input_seq, h_layer_seq, c_layer_seq, h_output_seq, c_output_seq],_ = theano.scan(	# frame number, depth, h_dimension
			step,
			sequences = x_seq,
			outputs_info = [h_in_init, c_in_init, h_init, c_init, h_out_init, c_out_init]
			)

		y_seq = self.softmax(h_output_seq)	# y_seq(frame, y_dim)

		cost = T.sum(np.absolute(h_output_seq[-1] - y_hat_seq))#T.sum((y_hat_seq * -T.log(y_seq)))

		dW,dB,dWf,dBf,dWi,dBi,dWo,dBo = [],[],[],[],[],[],[],[]
		for i in range(depth) :
			dW.append( T.grad(cost, self.Network[i].W) )
			dB.append( T.grad(cost, self.Network[i].B) )
			dWf.append( T.grad(cost, self.Network[i].Wf) )
			dBf.append( T.grad(cost, self.Network[i].Bf) )
			dWi.append( T.grad(cost, self.Network[i].Wi) )
			dBi.append( T.grad(cost, self.Network[i].Bi) )
			dWo.append( T.grad(cost, self.Network[i].Wo) )
			dBo.append( T.grad(cost, self.Network[i].Bo) )



		parameters = []
		gradients = []
		for i in range(depth):
			parameters.append(self.Network[i].W)
			parameters.append(self.Network[i].B)
			parameters.append(self.Network[i].Wi)
			parameters.append(self.Network[i].Bi)
			parameters.append(self.Network[i].Wo)
			parameters.append(self.Network[i].Bo)
			parameters.append(self.Network[i].Wf)
			parameters.append(self.Network[i].Bf)
			gradients.append(dW[i])
			gradients.append(dB[i])
			gradients.append(dWi[i])
			gradients.append(dBi[i])
			gradients.append(dWo[i])
			gradients.append(dBo[i])
			gradients.append(dWf[i])
			gradients.append(dBf[i])
		


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