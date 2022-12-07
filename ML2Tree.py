import pandas as pd
import numpy as np
import math
from random import randrange
import random

fullset=pd.read_csv("train.csv")
fullset.dtypes
see=fullset.describe()
#change some features to category type
listofcat=  ['area_cluster','make','segment','model','fuel_type','max_torque','max_power','engine_type','airbags','is_esc',
            'is_adjustable_steering','is_tpms','is_parking_sensors','is_parking_camera','rear_brakes_type',
            'cylinder','transmission_type','gear_box','steering_type',
            'is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
            'is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
for col in listofcat:
    fullset[col] = fullset[col].astype('category')
fullset["is_claim"]= fullset["is_claim"].astype('category')
see=fullset.describe()
listofnum=list(see.columns)
#one hot encoder for multitypes features, 1 for is_feature if yes
listofmulticat=  ['area_cluster','make','segment','model','fuel_type','max_torque','max_power','engine_type','airbags','rear_brakes_type',
            'cylinder','transmission_type','gear_box','steering_type']
listofboolcat=  ['is_esc', 'is_adjustable_steering','is_tpms','is_parking_sensors','is_parking_camera',
            'is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
            'is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
setmulticat=pd.get_dummies(fullset[listofmulticat])
setboolcat=fullset[listofboolcat].replace({'Yes': 1,'No': 0})
setnum=fullset[listofnum]
fullset=pd.concat([setmulticat, setboolcat, fullset[listofnum],fullset['is_claim']],axis=1)

def precision(predictionarray, label):
    try:
        TP = sum(np.logical_and(predictionarray == 1, label == 1))
        FP = sum(np.logical_and(predictionarray == 1, label == 0))
        return TP / (TP + FP)
    except Exception:
        return 0
def recall(predictionarray, label):
    try:
        TP = sum(np.logical_and(predictionarray == 1, label == 1))
        FN = sum(np.logical_and(predictionarray == 0, label == 1))
        return TP / (TP + FN)
    except Exception:
        return 0
def F1score(predictionarray, label):
    try:
        P=precision(predictionarray, label)
        R=recall(predictionarray, label)
        return 2 * P * R / (P + R)
    except Exception:
        return 0

#Tree component
class Node:
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, info_gain=None, value=None):
        # for decision node
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.info_gain = info_gain
        # for leaf node
        self.value = value

class Tree:
    def __init__(self, min_samples_split, max_depth):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def information_gain(self, parent, leftsplit, rightsplit):
        leftweight = len(leftsplit) / len(parent)
        rightweight = len(rightsplit) / len(parent)
        return self.entropy(parent) - (leftweight * self.entropy(leftsplit) + rightweight * self.entropy(rightsplit))

    def build_tree(self, X,Y, curr_depth=0, feature_size=1):
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(X,Y, feature_size)
            # check if information gain is positive then continue splitting
            if best_split.get("info_gain",0) > 0:
                # recur left
                leftsubtree = self.build_tree(best_split["dataset_left"][:, :-1],best_split["dataset_left"][:, -1], curr_depth + 1, feature_size)
                # recur right
                rightsubtree = self.build_tree(best_split["dataset_right"][:, :-1],best_split["dataset_right"][:, -1], curr_depth + 1, feature_size)
                # return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            leftsubtree, rightsubtree, best_split["info_gain"])

        # compute leaf node
        lY = list(Y)
        leaf_value = max(lY, key=lY.count)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, X, Y, feature_size):
        best_split = {}
        best_info_gain = -999
        n_rows, n_cols = X.shape
        # pick feature by random
        if feature_size != 1:
            n_feature = round(n_cols * feature_size)
            pickedfeature = np.sort(random.sample(range(n_cols), n_feature))
        else:
            pickedfeature = np.array(range(n_cols))
        # For every picked feature
        for feaidx in pickedfeature:
            X_curr = X[:, feaidx]
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, Y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[feaidx] <= threshold])
                df_right = np.array([row for row in df if row[feaidx] > threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
                    # Caclulate the information gain and save the split parameters
                    # if the current split if better then the previous best
                    gain = self.information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature': feaidx,
                            'threshold': threshold,
                            'dataset_left': df_left,
                            'dataset_right': df_right,
                            'info_gain': gain
                        }
                        best_info_gain = gain
        return best_split

    def fit(self, X, Y, feature_size=1):
        self.root = self.build_tree(X=X, Y=Y, feature_size=feature_size)

    def predictone(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        # Go to the left
        if feature_value <= tree.threshold:
            return self.predictone(x, tree.data_left)
        # Go to the right
        if feature_value > tree.threshold:
            return self.predictone(x, tree.data_right)

    def predictall(self, X):
        return [self.predictone(x, self.root) for x in X]

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        # report value
        if tree.value is not None:
            print(tree.value)
        # continue splitting
        else:
            print("X_" + str(tree.feature), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.data_left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.data_right, indent + indent)

dataset=fullset

Y=np.array(dataset['is_claim'].values).reshape(-1,1)
X=dataset[listofnum]

#binning to reduce computation cost
#binning boundaries
Xbin={}
for col in X.columns:
    ser, bins = pd.qcut(X[col], 100, retbins=True, labels=False,duplicates='drop')
    bins[0]=-999
    bins[-1]=999999
    Xbin[col]=bins
#binned data
Xbinned=[]
for col in X.columns:
    Xbinned.append(pd.cut(X[col], bins=Xbin[col], labels=False))
Xbinned = pd.concat(Xbinned, axis=1)

Xbinned=pd.concat([dataset.drop(listofnum,axis=1).drop('is_claim',axis=1),Xbinned],axis=1)
Xbinned=np.array(Xbinned)

'''
Treeclassifier = Tree(min_samples_split=10, max_depth=2)
Treeclassifier.fit(Xbinned,Y)
Treeclassifier.print_tree()
pred = np.array(Treeclassifier.predictall(Xbinned))

print(np.mean(pred==dataset['is_claim']))
print(precision(pred,dataset['is_claim']))
print(recall(pred,dataset['is_claim']))
print(F1score(pred,dataset['is_claim']))
'''

class RandomForest:
    def __init__(self, min_samples_split, max_depth, n_trees, sample_size, feature_size):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.feature_size = feature_size
        # Will store individually trained decision trees
        self.trees = []

    def subsample(self,dataset):
        sample = []
        n_sample = round(len(dataset) * self.sample_size)
        while len(sample) < n_sample:
            #picking by random
            Index = randrange(len(dataset))
            sample.append(dataset[Index])
        sample=np.array(sample)
        return sample[:, :-1], sample[:, -1].reshape(1,-1).T

    # Random Forest Algorithm
    def fit(self,datasetX,datasetY):
        # clear
        if len(self.trees) > 0:
            self.trees = []
        dataset=np.concatenate((datasetX, datasetY), axis=1)
        # build tree loop
        for i in range(self.n_trees):
            X,Y = self.subsample(dataset)
            Treeclassifier = Tree(round(self.min_samples_split * self.sample_size), self.max_depth)
            Treeclassifier.fit(X,Y,self.feature_size)
            tree=Treeclassifier
            self.trees.append(tree)

    def bagging_predict(self,x):
        # decided by most vote
        predictions = [tree.predictone(x,tree.root) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def predict(self,X):
        predictions = [self.bagging_predict(x) for x in X]
        return predictions

Forestclassifier=RandomForest(10,35,11,1,0.7)
Forestclassifier.fit(Xbinned,Y)
pred=np.array(Forestclassifier.predict(Xbinned))

print(np.mean(pred==dataset['is_claim']))
print(precision(pred,dataset['is_claim']))
print(recall(pred,dataset['is_claim']))
print(F1score(pred,dataset['is_claim']))

'''
compare with packages
'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=35, random_state=0,min_samples_split=10, class_weight='balanced_subsample')
model.fit(Xbinned, Y)
sk_pred=np.array(model.predict(Xbinned))
print(np.mean(sk_pred==dataset['is_claim']))
print(precision(sk_pred,dataset['is_claim']))
print(recall(sk_pred,dataset['is_claim']))
print(F1score(sk_pred,dataset['is_claim']))

from xgboost import XGBClassifier
model = XGBClassifier(scale_pos_weight=14.625, max_depth=35)
model.fit(Xbinned, Y)
xgb_pred=np.array(model.predict(Xbinned))
print(np.mean(xgb_pred==dataset['is_claim']))
print(precision(xgb_pred,dataset['is_claim']))
print(recall(xgb_pred,dataset['is_claim']))
print(F1score(xgb_pred,dataset['is_claim']))

from catboost import CatBoostClassifier
model = CatBoostClassifier(auto_class_weights='Balanced',depth=16,verbose=False,iterations=100)
model.fit(Xbinned, Y)
cat_pred=np.array(model.predict(Xbinned))
print(np.mean(cat_pred==dataset['is_claim']))
print(precision(cat_pred,dataset['is_claim']))
print(recall(cat_pred,dataset['is_claim']))
print(F1score(cat_pred,dataset['is_claim']))

from lightgbm import LGBMClassifier
model = LGBMClassifier(class_weight='balanced',max_depth=35)
model.fit(Xbinned, Y.ravel())
lgb_pred=np.array(model.predict(Xbinned))
print(np.mean(lgb_pred==dataset['is_claim']))
print(precision(lgb_pred,dataset['is_claim']))
print(recall(lgb_pred,dataset['is_claim']))
print(F1score(lgb_pred,dataset['is_claim']))

'''
part 2 cross validation and test
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xbinned, Y, test_size=0.1, random_state=0)
dataset = np.concatenate((X_train, y_train.reshape(1, -1).T), axis=1)
y_test=y_test.reshape(-1,)

# define fold number
Nfold=5
fullsize=dataset.shape[0]
size=round(fullsize/Nfold)
fold=[]
l = 0
u = size
# get fold range
for K in range(Nfold):
    fold.append(np.array(range(l,min(u,fullsize))))
    l+=size
    u+=size
# split dataset to train and validate and then run
CVForest=[]
for K in range(Nfold):
    validset = dataset[fold[K],:]
    NK = np.array([i for i in range(Nfold) if i != K])
    trainset = dataset[np.concatenate([fold[index] for index in NK]), :]
    Xt=trainset[:,:-1]
    Yt=trainset[:,-1].reshape(-1,1)
    Xv=validset[:,:-1]
    Yv=validset[:,-1].reshape(-1,)
    Forestclassifier = RandomForest(10, 35, 11, 1, 0.7)
    Forestclassifier.fit(Xt, Yt)
    CVForest.append(Forestclassifier)
    pred = np.array(Forestclassifier.predict(Xv))
    print(K)
    print(np.mean(pred == Yv))
    print(precision(pred, Yv))
    print(recall(pred, Yv))
    print(F1score(pred, Yv))

from sklearn import metrics
from sklearn.model_selection import cross_val_score

CVs=[3]
for CV in CVs:
    # run code above to make 'model'=selected model
    accuracy = cross_val_score(model, X_train, y_train, cv=CV, scoring='accuracy')
    precision = cross_val_score(model, X_train, y_train, cv=CV, scoring='precision')
    recall = cross_val_score(model, X_train, y_train, cv=CV, scoring='recall')
    F1score = cross_val_score(model, X_train, y_train, cv=CV, scoring='f1')
    print(accuracy,np.mean(accuracy))
    print(precision,np.mean(precision))
    print(recall,np.mean(recall))
    print(F1score,np.mean(F1score))

'''
testing
'''
Forestclassifier.fit(X_train, y_train)
pred=np.array(Forestclassifier.predict(X_test))
print(np.mean(pred==y_test))
print(precision(pred,y_test))
print(recall(pred,y_test))
print(F1score(pred,y_test))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=35, random_state=0,min_samples_split=10, class_weight='balanced_subsample')
model.fit(X_train, y_train)
sk_pred=np.array(model.predict(X_test))
print(np.mean(sk_pred==y_test))
print(precision(sk_pred,y_test))
print(recall(sk_pred,y_test))
print(F1score(sk_pred,y_test))

from xgboost import XGBClassifier
model = XGBClassifier(scale_pos_weight=14.625, max_depth=35)
model.fit(X_train, y_train)
xgb_pred=np.array(model.predict(X_test))
print(np.mean(xgb_pred==y_test))
print(precision(xgb_pred,y_test))
print(recall(xgb_pred,y_test))
print(F1score(xgb_pred,y_test))

from catboost import CatBoostClassifier
model = CatBoostClassifier(auto_class_weights='Balanced',depth=16,verbose=False,iterations=100)
model.fit(X_train, y_train)
cat_pred=np.array(model.predict(X_test))
print(np.mean(cat_pred==y_test))
print(precision(cat_pred,y_test))
print(recall(cat_pred,y_test))
print(F1score(cat_pred,y_test))

from lightgbm import LGBMClassifier
model = LGBMClassifier(is_unbalance='true',max_depth=35)
model.fit(X_train, y_train.ravel())
lgb_pred=np.array(model.predict(X_test))
print(np.mean(lgb_pred==y_test))
print(precision(lgb_pred,y_test))
print(recall(lgb_pred,y_test))
print(F1score(lgb_pred,y_test))

#create series of k fold cv classifiers
def createcvclass(Classifier,CVClassifier_num,X,Y):
    dataset = np.concatenate((X, Y.reshape(1, -1).T), axis=1)
    Nfold = CVClassifier_num
    fullsize = dataset.shape[0]
    size = round(fullsize / Nfold)
    fold = []
    l = 0
    u = size
    # get fold range
    for K in range(Nfold):
        fold.append(np.array(range(l, min(u, fullsize))))
        l += size
        u += size
    for K in range(CVClassifier_num):
        #validset = dataset[fold[K], :]
        NK = np.array([i for i in range(Nfold) if i != K])
        trainset = dataset[np.concatenate([fold[index] for index in NK]), :]
        Xt = trainset[:, :-1]
        Yt = trainset[:, -1].reshape(-1, 1)
        #Xv = validset[:, :-1]
        #Yv = validset[:, -1].reshape(-1, )
    CVClassifier = []
    for i in range(CVClassifier_num):
        ClassifierCom = Classifier
        ClassifierCom.fit(Xt,Yt)
        CVClassifier.append(ClassifierCom)
    return CVClassifier

def CVbagging_predict(CVClassifier,x):
    # decided by most vote
    x=x.reshape(1,-1)
    predictions = list(np.concatenate([fold.predict(x) for fold in CVClassifier], axis=0 ))
    return max(set(predictions), key=predictions.count)

def CVpredict(CVClassifier,X):
    predictions = [CVbagging_predict(CVClassifier,x) for x in X]
    return predictions

#pred=np.array(CVpredict(CVForest,X_test))
CVClassifier=createcvclass(model,5,X_train,y_train)
pred=np.array(CVpredict(CVClassifier,X_test))
print(np.mean(pred==y_test))
print(precision(pred,y_test))
print(recall(pred,y_test))
print(F1score(pred,y_test))
