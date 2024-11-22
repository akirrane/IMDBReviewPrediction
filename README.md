# IMDBReviewPrediction

This project is a attempt to derive sentiment from user reviews of movies, with the goal of being able to expand text analysis of movie reviews beyond platforms with inate scoring capabilities, giving movie studios the ability to use social media channels as direct feedback in a quick fashion without the need for human oversight.

## Running the code yourself

In order to run this program, it is best to create a virtual environment before downloading packages and libraries, my personal choice was Anaconda (I will be using Anaconda syntax for examples, syntax may vary across programs). Once you have created the environment and activated it, you will need to download the following libraries to run the main program.
```
conda install Flask, flask-cors, torch, torchvision, nltk
```
You will also need to activate python and download a specific nltk package.
```
$ python
>>> import nltk
>>> nltk.download('punkt')
```
Once this has been done, you can run the code yourself, starting with the train.py file. This will take in the dataset 'movieReviewsTrain.json' and train the neural net so that it is ready to classify (This should dump the saved model as model.pth).
```
>>> python train.py
```
From here you are able to run the predict.py file, which is a fully functioning terminal based application for the classifier.
```
>>> python predict.py
```
To run the model on the test data set, run the following command, which will classify the reviews ad display statistics.
```
>>> python test.py
```
If you want to utilize the frontend for a better user experience, run the app.py file to start the flask app in the background. 
```
>>> python app.py
```
You should be able to use the local host URL provided to open the UI, where you can test reviews to see the model's responses.
