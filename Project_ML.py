import sys;
import csv;
import statistics as sts;
import random as rnd;
import numpy as np;
from sklearn import svm
from sklearn import linear_model;
import os;

csv.register_dialect('spc', delimiter = ' ', escapechar = '\n');

def readfile(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f, dialect = 'spc');
        data = [];
        for r in reader:
            dt = [];
            for c in r:
                if c != '':
                    dt.append(int(c));
            data.append(dt);
        return data

def gettrainlabels(fname):
    Y = readfile(fname);
    L = [Y[i][0] for i in range(0, len(Y))]; 
    return L


# Output File for the testclass prediction and features
def writetestpred(testpred, indi, X_train15, X_val15, X_test15, acr):
    T = testpred;
    X = X_train15;
    XV = X_val15;
    XT = X_test15;
    fnum = len(indi);

    if os.path.exists('aa2897_Testclass_Prediction.txt'):
        os.remove('aa2897_Testclass_Prediction.txt');
    elif os.path.exists('aa2897_Features_Files.txt'):
        os.remove('aa2897_Features_Files.txt');

    pred = open('aa2897_Testclass_Prediction.txt', 'w+');
    
    for i in range(0, len(T)):
        pred.write('{} {}'.format(T[i], i));
        pred.write('\n');
    pred.close();

    fea = open('aa2897_Features_Files.txt', 'w+');
    
    fea.write('\nNumber of features used: {}\n'.format(fnum));
    fea.write('\nThe accuracy of this model is {}%\n'.format(acr));
    fea.write('\nFeatures Used in Model:---------------------------------------------------------------------------------------------\n');
    fea.write('\n');
    for i in range(0, len(indi)):
        fea.write('{} '.format(indi[i]));
    fea.close();

def chi_sqr(X, y, topf):
    rows = len(X);
    cols = len(X[0]);
    O = [];
    for j in range(0, cols):
        ctable = [[1,1],[1,1],[1,1]];
        for i in range(0, rows):
            if y[i] == 0:
                if X[i][j] == 0:
                    ctable[0][0] += 1
                elif X[i][j] == 1:
                    ctable[1][0] += 1
                elif X[i][j] == 2:
                    ctable[2][0] += 1
            elif y[i] == 1:
                if X[i][j] == 0:
                    ctable[0][1] += 1
                elif X[i][j] == 1:
                    ctable[1][1] += 1
                elif X[i][j] == 2:
                    ctable[2][1] += 1
        ctot = [sum(c) for c in ctable];
        rtot = [sum(c) for c in zip(*ctable)];
        tot = sum(ctot);
        exp = [[(r*c)/tot for r in rtot] for c in ctot];
        sqr = [[((ctable[i][j] - exp[i][j])**2)/exp[i][j] for j in range(0,len(exp[0]))] for i in range(0,len(exp))];
        chi2 = sum([sum(x) for x in zip(*sqr)])
        O.append(chi2);
    ind = sorted(range(len(O)), key=O.__getitem__, reverse=True); 
    indi= ind[:topf];
    return indi

def fextract(X, c):
    EF = [];
    col = list(zip(*X));
    for i in c:
        EF.append(col[i]);
    EF = list(zip(*EF));
    return EF

def frankselect(file_train_data, file_train_labels, file_test_data):
    X = readfile(file_train_data);
    Y = gettrainlabels(file_train_labels);
    Z = list(zip(Y,X));
    rnd.shuffle(Z);
    Y,X = zip(*Z);

    X = np.array(X); 
    Y = np.array(Y); 
	
    r1 = int(len(X) * 0.75);
    X_train = X[:r1];
    X_val = X[r1:];
    Y_train = Y[:r1];
    Y_val = Y[r1:];

    indi = chi_sqr(X_train, Y_train, 15)

    X_train15 = fextract(X_train, indi);
    X_train15 =np.array(X_train15); 

    X_val15 = fextract(X_val, indi);
    X_val15 = np.array(X_val15); 

    X_test = readfile(file_test_data);
    X_test = np.array(X_test); 
    X_test15 = fextract(X_test, indi);
    X_test15 = np.array(X_test15); 

    fnum = len(indi);
    print('\nNumber of features used: {}'.format(fnum))
    print('\nColumns used in prediction: {}'.format(indi))
	
    return X_train15, X_val15, X_test15, Y_train, Y_val, indi

def traintest(X_train15, X_val15, X_test15, Y_train, Y_val):
    linsvc = svm.LinearSVC(C=100.0,max_iter=100000);
    svc = svm.SVC(C=100.0,kernel='linear',tol=0.0001);
    logreg = linear_model.LogisticRegression(C=100.0,max_iter=100000);
    t1 = linsvc.fit(X_train15,Y_train);
    t2 = svc.fit(X_train15,Y_train);
    t3 = logreg.fit(X_train15,Y_train);

    tv1 = linsvc.predict(X_val15);
    tv2 = svc.predict(X_val15);
    tv3 = logreg.predict(X_val15);
    val_pred = np.array([]);
    for i in range(0,len(X_val15)):
        val_pred = np.append(val_pred, sts.mode([tv1[i],tv2[i],tv3[i]]))
    sc = 0;
    for i in range(0,len(Y_val)):
        if (val_pred[i] == Y_val[i]):
            sc += 1;
    acr = (sc/len(Y_val))*100;
    print('\nThe accuracy of this model is {}%'.format(acr));

    tt1 = linsvc.predict(X_test15);
    tt2 = svc.predict(X_test15);
    tt3 = logreg.predict(X_test15);
    test_pred = np.array([]);
    for i in range(0,len(X_test15)):
        test_pred = np.append(val_pred, sts.mode([tt1[i],tt2[i],tt3[i]]))
    return test_pred, acr

if __name__ == '__main__':
    trainfile = sys.argv[1];
    tlabelfile = sys.argv[2];
    testfile = sys.argv[3];
    

 
    X_train15, X_val15, X_test15, Y_train, Y_val, indi = frankselect(trainfile, tlabelfile, testfile);

    test_pred, acr = traintest(X_train15, X_val15, X_test15, Y_train, Y_val);

    writetestpred(test_pred, indi, X_train15, X_val15, X_test15, acr);
