# NeuralNetwork
NeuralNetworks

 # ABSTRACT
A Neural Network is used for data mining and analysis. It it an intelligent computer program that can detect patterns and changes in a data-set. For the purpose of stock market prediction, it uses Time Series prediction methods to analyze the patterns and changes in the data overtime. The type of neural network used in this paper is a back propagation multi-layer perceptron (MLP) network. This paper shall detail the process of collating data and applying it to the network to see if the network is capable of predicting the stock price. The network gave varying results but always hovered around a Mean Sum of Squared Errors of around 0.08…. thus showing potential.
# 1.	INTRODUCTION 
Attempting to predict the value of stock prices is not a new concept. People have tried numerous methods for decades. There are hundreds of techniques for market prediction. Most notable is the idea of Quant analysis, used by the large investment banks throughout the world. 
They rely on the principle of Time Series Prediction. The idea of being able to calculate future data based on the analysis of trends and patterns within data. The principle is based on probability, for example if you flipped a coin 50 times and it landed face up (heads) 39 times odds are you would pick heads guessing/making an assumption that it was most likely to land on heads again. Time series prediction attempts to apply this principle to larger groups of data. 
For the purpose of stock market prices using an MLP, the idea is that given enough data and an adequate learning rate and momentum, the network will be able to predict the mean price for the next day of trading. Thus predicting the financial crash of 2008 and by feeding it data from 2006 - 2007 and comparing the predicted numbers of Mean Sum of Squared Errors of the network's outputs to the actual numbers available to us we can see it has learnt.
The Network works on the principle of Sigmoidal Neurons and the back propagation algorithm, it takes a series of inputs and applies an associated weight, then passes the inputs to a series of hidden neurons and does the weighted sum. The process then repeats until the data has been "learnt". This is achieved through object orientated program using C++ and is developed in such a way that simple yet refined learning algorithms are hard coded into the very framework of the system.  

# 2.	APPLICATION OF DATA
For this task, the problem was about experimenting with the neural networks ability to "learn" certain types of problems and its inability to learn others. This presented an interesting opportunity, because of the way the network learns, it is able to be applied to "TSP" and allowed for an experiment on Stock Market data. Using historic stock prices, in sets of open, high, low and close values (e.g. O = 400, H = 600, L = 250, C = 500) I was able to apply the idea to 3 inputs and 1 output variable. With the close being the associated output for the particular set. The data sets are the stock prices for each day during the period 2006 - 2007. 
Applying stock market data to a neural network is commonly done in investment banks to give them an edge on the stock market when making trading decisions. The thought was to take an already established network and highly analyzed data and attempt to see if it can 'learn' trends in financial data and test its ability to spot declines. Thus in principle it would allow for predictions of crashes and would help to alleviate some of the fallout from such an event. 
 
Figure 1. Stock Market Graph



# 2.1.	Reasoning Behind Data
One of the things that needed to be considered was the idea that any problem applied to the network would need to fit within the constraints of the system. So we had to pick a problem that could easily be applied to the network, otherwise major changes and alterations to the network would need to be made. so for this, data was sought that could be sorted and formatted to the desired template and then simply called in the program and the tests applied. 
All the data for the tests was sourced from the Yahoo Finance archive online and was then converted into the desired format and then split into three files train.txt. unseen.txt and valid.txt, the ratio for splitting the data was based on a genetic algorithm utilizing a ratio of 30 sets for train, 15 sets for valid and 5 sets for unseen this out of 533 sets of data. giving values of 326 for train, 53 for unseen and 154 for valid. 
This ratio proved to be better than simply taking it in larger chunks. We initially started off by simply taking the first 310 inputs at train and then the next 153 as valid and the rest as unseen, while the ratio was ok the splitting method was poor, showing that smaller ratios are better, as an experiment done later on with a smaller data set but with a ratio of 10, 5 and 2 provided a much smaller sum of squared errors. Even getting down to as little as 0.008 for one of the tests. The genetic algorithm can be found here - http://people.cs.pitt.edu/~hashemi/papers/CISIM2010_HBHashemi.pdf
 
# 2.2.	Testing The Concept
After sorting through the the data and formatting and splitting the data had been done, I needed to ensure the network could read the files, luckily DR Mitchell had already provided a method for reading in files for the Numerical problem. This allowed me to simply change the file names in the code and copy the .text files into the program folder. 
Next comes applying the tests to the data and checking if the network can learn the stock market. 
For my initial tests I went with the default weights and 5 hidden neurons, 500 epochs and validation on with a learning rate of 0.7 and momentum of 0.8 I kept these settings for for tests of test changing only the learn rate and momentum for 5 tests. 
This provided a mean SSE of around 0.3 and severally limited the networks ability to learn. So the number of hidden neurons was changed to 10 for the next 10 tests. 
 
Figure 2. Tadpole Plot Graph 

After this the number of hidden neurons was then increased to 15 for the nest 20 tests and the learn rate and momentum were altered to lower and lower numbers including 0.009. this provided a much better Mean SSE of around 0.088 - 0.0814 

 

Figure 3. Output from Console.
 
Figure 4. Output from Console.
Eventually after increasing the number of neurons in blocks of around 5 for a long time I ended up with 200 neurons and the default Picton weights and momentum and learn rates changing with every test. 
 
Figure 4. Output from Console 2.

# 2.3.	What are the Results
So after over 100 different experiments with varying learning rates and momentum values and hidden neurons ranging from 5 - 200 and the weights set to default the network provided an estimated output of around 5133 according to validfull.txt and a lowest a mean sum of squared errors of around 0.08001. the network consistently stopped learning between 180 - 210 epochs. 

# 2.4.	Discussion of Results
The networks ability to learn is on show as the SSE drops lower, this clearly demonstrates the networks ability to at least partially learn patterns in complex data. 
The lowest the SSE ever got to was 0.08001 indicating that although the network appears to be learning the problem partially it is clearly lacking. 
It clearly indicates a strong foundation to start building off of, but the results are a lot less than ideal.
They are not as accurate as anticipated/desired and have been unable to empirically prove weather or not the network is capable or predicting a financial crash in the stock market.
 The tadpole graphs show a slow depression/decline in the price of the stocks but doesn't provide enough detail to be a true indicator of a potential crash. 

 
# 3.	TO CONCLUDE
The experimented demonstrated that in principle neural networks have the ability to recognize patterns in large sets of data and can be used to solve complex problems by data analysis. 
While the network couldn't indicate with 100% certainty that there would be a crash in 2008, given that we already know it happened the networks limited but clear results show that with enough data and time, and a well defined ratio the program would indicated potential falls in stock prices and maybe even allow economists to more carefully monitor the price and develop methods for limiting the potential fallout of future collapses. Providing a reference point for future work to be carried out on the topic. 
