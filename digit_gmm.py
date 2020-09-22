# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:44:27 2020

@author: Yamini
"""

import glob
import numpy as np
import librosa
from sklearn import mixture

files = glob.glob('./train/*.wav')
fs = 16000  #sampling frequency = 16 kHz
w_s = int(fs * 0.025)       #window length 25 ms
hop_length = int(w_s /2.5)      #window shift 10 ms
coef0 = np.empty((0,39))
coef1 = np.empty((0,39))
coef2 = np.empty((0,39))
coef3 = np.empty((0,39))
coef4 = np.empty((0,39))
coef5 = np.empty((0,39))
coef6 = np.empty((0,39))
coef7 = np.empty((0,39))
coef8 = np.empty((0,39))
coef9 = np.empty((0,39))
 
print("Training data collecting...")
for i in files:
    label =int(i.split('_')[1]) #digit labels
    datas,fss = librosa.load(i,sr = fs)
    data,index = librosa.effects.trim(datas,top_db=20)   #trim the audio for db<20
    
    #mfcc,delta and delta delta components taken. 39 dimentional features
    
    coef = librosa.feature.mfcc(y=data, sr=fs,n_mfcc = 13,hop_length = hop_length,n_fft = w_s ).T
    delta = librosa.feature.delta(coef)
    deldel = librosa.feature.delta(delta)
    coef = np.append(coef,delta,axis = 1)
    coef = np.append(coef,deldel,axis = 1)
    
    if label == 0:                                  #splitting features for individual digits
        coef0 = np.append(coef0,coef,axis = 0)
    elif label == 1:
        coef1 = np.append(coef1,coef,axis = 0)
    elif label == 2:
        coef2 = np.append(coef2,coef,axis = 0)
    elif label == 3:
        coef3 = np.append(coef3,coef,axis = 0)
    elif label == 4:
        coef4 = np.append(coef4,coef,axis = 0)     
    elif label == 5:
        coef5 = np.append(coef5,coef,axis = 0)
    elif label == 6:
        coef6 = np.append(coef6,coef,axis = 0)
    elif label == 7:
        coef7 = np.append(coef7,coef,axis = 0)
    elif label == 8:
        coef8 = np.append(coef8,coef,axis = 0) 
    elif label == 9:
        coef9 = np.append(coef9,coef,axis = 0)
    else:
        print("invalid")

print("Training data collected. Starting fitting GMMS...")  #GMM initialization and fitting

g0 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 0...")
g0.fit(coef0)  
print("GMM fitted for 0.")
g1 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 1...")
g1.fit(coef1)
print("GMM fitted for 1.")  
g2 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 2...")
g2.fit(coef2)  
print("GMM fitted for 2.")
g3 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 3...")
g3.fit(coef3)  
print("GMM fitted for 3.")
g4 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 4...")
g4.fit(coef4)  
print("GMM fitted for 4.")
g5 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 5...")
g5.fit(coef5)  
print("GMM fitted for 5.")
g6 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 6...")
g6.fit(coef6)  
print("GMM fitted for 6.")
g7 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 7...")
g7.fit(coef7)    
print("GMM fitted for 7.")
g8 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 8...")
g8.fit(coef8)  
print("GMM fitted for 8.")
g9 = mixture.GaussianMixture(n_components=16)
print("GMM initialized for 9...")
g9.fit(coef9) 
print("GMM fitted for 9.")

print("Testing starts...")  #testing
test_files = glob.glob('./test/*.wav')
conf = np.zeros((10,10))   

for j in test_files:
    label = int(j.split('_')[1])
    datas,fss = librosa.load(j,sr = fs)
    data,index = librosa.effects.trim(datas,top_db=36)  #trim the audio for db<36
    coef = librosa.feature.mfcc(y=data, sr=fs,n_mfcc = 13,hop_length = hop_length,n_fft = w_s ).T
    delta = librosa.feature.delta(coef)
    deldel = librosa.feature.delta(delta)
    coef = np.append(coef,delta,axis = 1)
    coef = np.append(coef,deldel,axis = 1)
    likelihood = np.array([g0.score(coef),g1.score(coef),g2.score(coef),g3.score(coef),g4.score(coef),g5.score(coef),g6.score(coef),g7.score(coef),g8.score(coef),g9.score(coef)])
    #confusion matrix using max likelihood
    conf[label,likelihood.argmax()] = conf[label,likelihood.argmax()] +1    
print("Testing completed.")

print(conf) #confusion matrix