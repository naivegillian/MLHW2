import pandas as pd
import numpy as np
import math
import random
fullset=pd.read_csv("train.csv")
#reduce classes of displacement
fullset['displacement']=np.floor(fullset['displacement']/10)
fullset.dtypes
see=fullset.describe()
#change some features to category type
listofcat=  ['area_cluster','make','segment','model','fuel_type','max_torque','max_power','engine_type','airbags','is_esc',
            'is_adjustable_steering','is_tpms','is_parking_sensors','is_parking_camera','rear_brakes_type','displacement',
            'cylinder','transmission_type','gear_box','steering_type',
            'is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
            'is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
for col in listofcat:
    fullset[col] = fullset[col].astype('category')
fullset["is_claim"]= fullset["is_claim"].astype('category')

#performance assessment
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

ransort = random.sample(range(fullset.shape[0]), round(fullset.shape[0]*0.9))
dataset = fullset.iloc[ransort,:]
testset = fullset.drop(list(ransort), axis=0, inplace=False)
fullsize=dataset.shape[0]

# define fold number
Nfold=3
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
for K in range(Nfold):
    validset = dataset.iloc[fold[K],:]
    NK = np.array([i for i in range(Nfold) if i != K])
    trainset = dataset.iloc[np.concatenate([fold[index] for index in NK]), :]

    see=trainset.describe()
    #generate prior
    Pis0,Pis1 = trainset["is_claim"].value_counts()
    Prob0 = Pis0/(Pis0+Pis1)
    Prob1 = Pis1/(Pis0+Pis1)

    #generate likelihood of each categorical feature
    d0= {}
    d1= {}
    for col in listofcat:
        total = trainset[col].value_counts()
        given0=trainset[trainset["is_claim"]==0][col].value_counts()
        given1=trainset[trainset["is_claim"]==1][col].value_counts()
        d0[col]=given0/Pis0
        d1[col]=given1/Pis1

    #generate likelihood of each numerical feature
    see0=trainset[trainset["is_claim"]==0].describe()
    see1=trainset[trainset["is_claim"]==1].describe()

    def calculate_probability(x, mean, stdev):
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * math.exp(-((x-mean)**2 / (2 * stdev**2 )))

    listofnum=see.columns
    listofnum=list(listofnum[np.isin(listofnum,listofcat)==False])

    #compute posterior
    pred=[]
    for i in range(validset.shape[0]):
        sample = validset.iloc[i, :]
        l0 = 1
        l1 = 1
        for col in listofnum:
            l0=l0*(calculate_probability(sample[col],see0[col]['mean'],see0[col]['std']))
            l1=l1*(calculate_probability(sample[col],see1[col]['mean'],see1[col]['std']))
        for col in listofcat:
            l0=l0*d0.get(col).get(sample[col])
            l1=l1*d1.get(col).get(sample[col])
        if(l0*Prob0>l1*Prob1):
            pred.append(0)
        else:
            pred.append(1)

    pred=np.array(pred)
    print(K)
    print(np.mean(pred==validset['is_claim']))
    print(precision(pred,validset['is_claim']))
    print(recall(pred,validset['is_claim']))
    print(F1score(pred,validset['is_claim']))

#test
see=dataset.describe()
#generate prior
Pis0,Pis1 = dataset["is_claim"].value_counts()
Prob0 = Pis0/(Pis0+Pis1)
Prob1 = Pis1/(Pis0+Pis1)

#generate likelihood of each categorical feature
d0= {}
d1= {}
for col in listofcat:
    total = dataset[col].value_counts()
    given0=dataset[dataset["is_claim"]==0][col].value_counts()
    given1=dataset[dataset["is_claim"]==1][col].value_counts()
    d0[col]=given0/Pis0
    d1[col]=given1/Pis1

#generate likelihood of each numerical feature
see0=dataset[dataset["is_claim"]==0].describe()
see1=dataset[dataset["is_claim"]==1].describe()

def calculate_probability(x, mean, stdev):
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * math.exp(-((x-mean)**2 / (2 * stdev**2 )))

listofnum=see.columns
listofnum=list(listofnum[np.isin(listofnum,listofcat)==False])

#compute posterior
pred=[]
for i in range(testset.shape[0]):
    sample = testset.iloc[i, :]
    l0 = 1
    l1 = 1
    for col in listofnum:
        l0=l0*(calculate_probability(sample[col],see0[col]['mean'],see0[col]['std']))
        l1=l1*(calculate_probability(sample[col],see1[col]['mean'],see1[col]['std']))
    for col in listofcat:
        l0=l0*d0.get(col).get(sample[col])
        l1=l1*d1.get(col).get(sample[col])
    if(l0*Prob0>l1*Prob1):
        pred.append(0)
    else:
        pred.append(1)
print(np.mean(pred==testset['is_claim']))
print(precision(pred,testset['is_claim']))
print(recall(pred,testset['is_claim']))
print(F1score(pred,testset['is_claim']))