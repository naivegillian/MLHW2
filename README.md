# MLHW2

Naive Bayes Code in ML2Naive.py
Tree Based Classifier Code in ML2Tree.py

Each py file contains several parts, please open them with IDE and run the codes by parts
------
ML2Naive.py

row 1 to 46:
import package and data
preprocess
split data into training/validating and testing

row 48 to 113:
split training/validating into 3 folds
cross validation 
  compute prior, likelihood and posterior with training set
  print performance on validation set
  
row 115 to 161:
compute prior, likelihood and posterior with training set
print performance on testing set

------
ML2Tree.py

row 1 to 30:
import package and data
preprocessing: transform categories/boolean in numbers

row 32 to 52:
performance assessment function

row 54 to 177:
decision tree classifier

row 179 to 199:
preprocessing: bin numeric data

row 201 to 211:
trial of decision tree

row 213 to 254:
decision tree classifier

row 256 to 263:
fit and see performance of random forest, use all the samples as training data

row 265 to 302:
try public packages, use all the samples as training data

row 304 to 310:
9:1 train test split

row 312 to 341:
cross-validation of self made random forest, with output a classifier

row 344 to 357:
cross-validation performance assessment of public package classifiers
please go to row 265 to 302 to make model=model to be tested

row 359 to 403:
try public packages, train with training set and test with testing set

row 405 to 450:
an algorithm to generate k fold cv classifier
test it
please go to row 265 to 302 to make model=model to be tested
