# Simple-Autism-Prediction-Algorithm
A simple classification decision tree algorithm which predicts if someone has Autistic Spectrum Disorder (ASD). The original dataset can be found at: https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Autism%20Screening%20Adult 

The files consist of the orignaal dataset, two cleaned datasets and two python files. Data Processing file was used to clean the data to remove any rows with missing or null values and to remove any anoamlies, creating the cleaned data file. Since the data is categorical, I encoded it using onehotencoder so that it could be used to build the algorithm.

The main algorithm is created using sklearn decision tree classifier achieving an accuracy of 89.0%.
