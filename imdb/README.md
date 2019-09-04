# Summary:
Here we will create classifier for IMDB using CNN 
The NN has been trained via Google colab platform over GPU's

The problem that we will use to demonstrate sequence learning in this tutorial is the IMDB movie review sentiment classification problem. Each movie review is a variable sequence of words and the sentiment of each movie review must be classified.
The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given movie review has a positive or negative sentiment.
The data was collected by Stanford researchers and was used in a 2011 paper where a split of 50-50 of the data was used for training and test. An accuracy of 88.89% was achieved.
Keras provides access to the IMDB dataset built-in. The imdb.load_data() function allows you to load the dataset in a format that is ready for use in neural network and deep learning models.
The words have been replaced by integers that indicate the ordered frequency of each word in the dataset. The sentences in each review are therefore comprised of a sequence of integers.

 

# Installations:
* ANALYZE: Pandas, numpy, jupyter
* ML: tensorflow, keras [Installation guide](https://www.dataweekends.com/blog/2017/03/09/set-up-your-mac-for-deep-learning-with-python-keras-and-tensorflow)
* Testing: Pillow

# Data sources:
* [Google Drive presentation - trying to improve NN (CIFAR10_CNN, IMDB_CNN, IMDB_LSTM)](https://docs.google.com/presentation/d/1lkY-S3NGL3Q42Jcx6vovSukN4HTQ9x4PTnBnekvtAOc/edit#slide=id.g4b729b8d53_0_0)
