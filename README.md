# Twitter User Gender Classification (NLP, Tensorflow)
Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.TensorFlow 2.0 is the latest version of Google's flagship deep learning platform.TensorFlow 2.0 uses Keras API as its default library for training classification and regression models. 

In this notebook, I am going to implement NLP using Tensorflow to understand whether the post/description written on Twitter is written by a man or woman.
 The dataset contains 20,000 rows, each with a user name, a random tweet, account profile and image, location, and even link and sidebar color. Iam only using the description column to predict the gender.Thus the dependant feature being Gender, Description is the Independant feature.
 
The dataset can be downloaded from here:https://www.kaggle.com/crowdflower/twitter-user-gender-classification
 
 Cleaning/EDA:  
 I have removed the unnecessary categories like id, retweet count, location etc. and taken the description column.
I have also removed the null values and  used the Regular Expression library to clean my data (Regular Expression Library is using for searching a pattern).
In this part, I cleaned all irrelevant words. For example, if we have a sentence like: "I go to the school every day." we don't need some words ("the", "to" etc.) while classifying if a sentence was written by a male or female. So we got rid of them.
The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
So, using the bag of words method, we got 5000 most used words thereby cleaning the data making it easy for our model. I have also trained the same model without using the re library. But it is clear that the accuracy decreases considerably as you can see in the respective notebooks.
After the cleaning, I have also plotted a pie chart (using matplotlib plot(kind=pie)) and a Bar Plot (data.gender.value_counts().plot.bar())
 for the gender column which shows male/female/unknown/brand.
Also, the gender column also contained values ‘brand’ and ‘unknown’. I have removed them and converted it ‘male’ and ‘female’ to 0 and 1 for sake of simplicity. Then again I have analysed the data (now 0 and 1) using a bar plot.

Activation Functions:   
Using TensorFlow ,I have chosen ‘RELU’ and ‘SOFTMAX’ as the activation function.
Advantages of ‘relu’ and ‘softmax’ functions:
The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time. This means that the neurons will only be deactivated if the output of the linear transformation is less than 0.
Relu behaves close to a linear unit. Thus, it's a very simple function. That makes optimisation much easier.
The softmax activation function is used in neural networks when we want to build a multi-class classifier which solves the problem of assigning an instance to one class when the number of possible classes.It is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.

Optimizer:   
The model is trained using categorical_crossentropy loss function and adam optimizer. The evaluation metric is accuracy.
A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

Neural Network used:   
ANN (ARTIFICIAL NEURAL NETWORK) is used in this prediction model. The reason being that the particular problem is a CLASSIFICATION PROBLEM.
ANN is generally the best for classification and mainly multi class classifications as compared to CNN which is used generally for only Image Processing.
I have used the Sequential model and run the model for 50 epochs. The validation set being 20%. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
For my model, I have used 3 dense layers for the model, the last one being the output layer.

