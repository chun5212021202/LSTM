# LSTM

1. Raw dataset is from MCTest- machine comprehension test. Check it out if you are interested (http://research.microsoft.com/en-us/um/redmond/projects/mctest/)

2. The 4 json files are Word2Vec data trained by GloVe, and they're what we use for training and validation.

3. Run with command "python mainLSTM.py", but once again, check the data-file-path

4. Parameters setting of the network, such as width or depth, can be configured in mainLSTM.py

5. There are several LSTM model, including LSTM.py, LSTM_1layer.py, LSTM_3layer.py; remember to import the one you actually want into mainLSTM.py, and set LSTM_DEPTH to the correct value.

6. As the file names, LSTM_1layer.py and LSTM_3layer.py are 1-layer-LSTM and 3-layer-LSTM. And LSTM.py is supposed to be "costumized layer LSTM", but I recently didn't get enough time working on it, and therefore it doesn't really work fine. 

7. Default parameters are as follows. You are strongly suggested to adjust those values properly, since they aren't quite robust.
        
      LSTM_EPOCH = 20

      LSTM_DEPTH = 1

      LSTM_H_DIMENSION = 512
      
      LSTM_X_DIMENSION = 300
      
      LSTM_Y_DIMENSION = 300
      
      LSTM_LEARNING_RATE = 0.001

      LSTM_DECAY = 0.99994  #NOT USED! SO IGNORE IT
      
      LSTM_ALPHA = 0.99
      
      LSTM_GRAD_BOUND = 0.1 #NOT USED! SO IGNORE IT
      
      LSTM_OUTPUT_FILE = 'ResultLSTM/result_test.lab'

8. Optimizations should be implemented in LSTM_XXX.py or LSTM.py. You can use different activation function or optimizer, the default is RMSProp.

9. The training results might not be so good, because the whole learning model I designed for MCTest is supposed to be the merge of DNN and LSTM. Here it is just a simple demo of LSTM alone. But still, you'll find out how strong LSTM is compared to RNN  
