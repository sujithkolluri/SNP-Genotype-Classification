In this course project we encourage you to develop your own set of methods 
for learning and classifying. 

We will test your program on the dataset provided for the project. This is 
a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise.

In the training are 4000 cases and 4000 controls. Your task is to predict 
the labels of 2000 test individuals whose true labels are known only to 
the instructor and TA. 

Both datasets and labels are immediately following the link for this
project file. The training dataset is called traindata.gz (in gzipped
format), training labels are in trueclass, and test dataset is called
testdata.gz (also in gzipped format).

You may use cross-validation to evaluate the accuracy of your method and for 
parameter estimation. 

Your project must be in Python. You cannot use numpy or scipy except for numpy 
arrays as given below. You may use the support vector machine, logistic regression, 
naive bayes, linear regression and dimensionality reduction modules but not the 
feature selection ones. These classes are available by importing the respective 
module. For example to use svm we do

from sklearn import svm

You may also make system calls to external C programs for classification
such as svmlight, liblinear, fest, and bmrm.