#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from operator import add

# Implémentation de différents algorithme pour classifier des données en deux classes

def normalize(a) :
    size = a.shape[1]
    maxs = a.max(axis=0)
    mins = a.min(axis=0)
    np.append(maxs,[1])
    print mins
    normalized = (a-mins)/(maxs-mins)
    print (normalized > 1).sum()
    print (normalized < 0).sum()
    return normalized

# Tout d'abord nous allons modifier le fichier iris.data de sorte que les iris-setosa appartiennent à la classe +1
# et que les autres à la classe -1
def getIris () :
    fname = "iris.data"
    f = open(fname, "r")
    lines = []
    for line in f :
        line = line.strip("\n")
        line_tab = line.split(",")
        if line_tab[4] == "Iris-setosa" :
            line_tab[4] = "+1"
        else :
            line_tab[4] = "-1"
        lines.append(line_tab)
    mon_set =  np.array(lines, dtype=float) # Convertir le contenu du tableau en float
    return normalize(mon_set)

def getCancer () :
    fname = "cancer.data"
    f = open(fname, "r")
    lines = []
    for line in f :
        line = line.strip("\n")
        line_tab = line.split(",")
        res = line_tab[2:] + [0]
        if line_tab[1] == "M" :
            res[len(res)-1] = "+1"
        else :
            res[len(res)-1] = "-1"
        lines.append(res)
    mon_set = np.array(lines, dtype=float) # Convertir le contenu du tableau en float
    return normalize(mon_set)

def getMushroom() :
    fname = "agaricus-lepiota.data"
    f = open(fname, "r")
    lines = []
    for line in f :
        line = line.strip("\n")
        line_tab = line.split(",")
        res = line_tab[1:] + ["e"]
        res = map(lambda x : (ord(x) - 96)/33., res)
        if line_tab[0] == "e" :
            res[len(res)-1] = "+1"
        else :
            res[len(res)-1] = "-1"
        lines.append(res)
    mon_set = np.array(lines, dtype=float) # Convertir le contenu du tableau en float
    return mon_set

def getSpamBase() :
    fname = "spambase.data"
    f = open(fname, "r")
    lines = []
    for line in f :
        line = line.strip("\n")
        line_tab = line.split(",")
        if int(line_tab[len(line_tab)-1]) == 1 :
            line_tab[len(line_tab)-1] = "+1"
        else :
            line_tab[len(line_tab)-1] = "-1"
        lines.append(line_tab)
    mon_set = np.array(lines, dtype=float) # Convertir le contenu du tableau en float
    return normalize(mon_set)

def getWine () :
    fname = "wine.data"
    f = open(fname, "r")
    lines = []
    for line in f :
        line = line.strip("\n")
        line_tab = line.split(",")
        line_tab = line_tab[1:]
        res = line_tab[1:] + [0]
        if line_tab[0] == "1" :
            res[len(res)-1] = "+1"
        else :
            res[len(res)-1] = "-1"
        lines.append(res)
    return np.array(lines, dtype=float) # Convertir le contenu du tableau en float



# S la base d'apprentissage
# T le nombre max d'itérations
# eta le pas d'apprentissage
def perceptron (S, T, eta) :
    size_of_space = len(S[0]) -1
    w = [0] * (size_of_space + 1)
    for t in range(0,T) :
        i = random.randint(0,len(S)-1)
        yi = S[i][size_of_space]
        xi = [1] + list(S[i][0:size_of_space])
        if  (yi * np.dot(w, xi)) <= 0 :
            w = map(add, w, [i * eta * yi for i in xi])
    return w


# ADALINE

def hw(w,x) :
    return w[0] + np.dot(w[1:], x)

def L_adaline(w,S) :
    size_of_space = len(S[0])-1
    m = len(S)
    somme = 0
    for i in range(1,m) :
        xi = S[i][0:size_of_space]
        yi = S[i][size_of_space]
        somme = somme + pow(hw(w,xi) - yi ,2)
    return somme/m

def initw(size) :
    w = [0.] * size
    for i in range(1,size) :
        w[i] = random.randint(0,100)/100.
    return w

# S la base d'apprentissage
# T le nombre maximum d'itérations
# eta le pas d'apprentissage
# E la précision
def adaline(S, T, eta, E) :
    size_of_space = len(S[0]) -1
    w = initw(size_of_space+1)
    t = 0
    condition = True
    while condition :
        # Choisir un exemple au hasard
        i = random.randint(0,len(S)-1)
        yi = S[i][size_of_space]
        xi = S[i][0:size_of_space]
        #Mettre à jour les poids
        w2 = [0] * (size_of_space + 1)
        w2[0] = w[0] - 2*eta*(hw(w, xi)-yi)
        w2[1:] = w[1:] - 2*eta*xi*(hw(w, xi)-yi)
        t = t+1
        condition = t < T and abs(L_adaline(w2, S) - L_adaline(w, S)) > E
        w = w2
    return w

def sigma(x) :
    return 1 / (1 + np.exp(-x))

def L_logistique(w,S) :
    size_of_space = len(S[0])-1
    m = len(S)
    somme = 0
    for i in range(1,m) :
        xi = S[i][0:size_of_space]
        yi = S[i][size_of_space]
        somme = somme + np.log(1 + np.exp(-yi * hw(w,xi)))
    return somme/m

# S la base d'apprentissage
# T le nombre maximum d'itérations
# eta le pas d'apprentissage
# E la précision
def logistique(S, T, eta, E) :
    size_of_space = len(S[0]) -1
    w = initw(size_of_space+1)
    t = 0
    condition = True
    while condition :
        # Choisir un exemple au hasard
        i = random.randint(0,len(S)-1)
        yi = S[i][size_of_space]
        xi = S[i][0:size_of_space]
        #Mettre à jour les poids
        w2 = [0] * (size_of_space + 1)
        w2[0] = w[0] - eta * ( -yi * (1 - sigma(hw(w, xi))))
        w2[1:] = w[1:] - eta * (-yi * xi * (1 - sigma(hw(w, xi))))
        t = t+1
        condition = t < T and abs(L_logistique(w2, S) - L_logistique(w, S)) > E
        w = w2
    return w

def L_exponentielle(w,S) :
    size_of_space = len(S[0])-1
    m = len(S)
    somme = 0
    for i in range(1,m) :
        xi = S[i][0:size_of_space]
        yi = S[i][size_of_space]
        somme = somme + np.exp(-yi * hw(w,xi))
    return somme/m

# S la base d'apprentissage
# T le nombre maximum d'itérations
# eta le pas d'apprentissage
# E la précision
def exponentielle(S, T, eta, E) :
    size_of_space = len(S[0]) -1
    w = initw(size_of_space+1)
    t = 0
    condition = True
    while condition :
        # Choisir un exemple au hasard
        i = random.randint(0,len(S)-1)
        yi = S[i][size_of_space]
        xi = S[i][0:size_of_space]
        #Mettre à jour les poids
        hwx = hw(w, xi)
        w2 = [0] * (size_of_space + 1)
        w2[0] = w[0] - eta * (-yi * np.exp(-yi * hwx))
        w2[1:] = w[1:] - eta * (-yi * xi * np.exp(-yi * hwx))
        t = t+1
        condition = t < T and abs(L_exponentielle(w2, S) - L_exponentielle(w, S)) > E
        w = w2
    return w

# BT la base de test
# wT le résultat
def testMethode(BT, wT) :
    size_of_space = len(BT[0]) -1
    perf = 0.0
    for i in range(0,len(BT)) :
        yi = BT[i][size_of_space]
        xi = [1] + list(BT[i][0:size_of_space])
        if (yi * np.dot(wT, xi)) > 0 :
            perf += 1.0
    return perf/float(len(BT))

# S le set de données
# k le nombre de sous-ensembles sur lesquels faire la cross-validation
# T le nombre de pas
def choixEta(S,k,T) :
    size = len(S)
    eta_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    res = np.zeros((7,1))
    data = np.array((k,2))
    cut_size = size / k

    for step in range(0,k) :
        cut = step * cut_size
        test_list = S[cut:cut+cut_size]
        learn_list = np.vstack((S[0:cut], S[cut+cut_size:size])) #vstack concatenates the arrays of arrays
        i = 0
        for eta in eta_range :
            wT = perceptron(learn_list,T,eta)
            res[i] += testMethode(test_list,wT)
            i += 1

    best_perf = max(res)
    i = 0
    retour = []
    for perf in res :
        if perf == best_perf :
            retour = retour + [eta_range[i]]
        i += 1
    print "Résultat pour eta = " + str(retour)



# Récupération des données, mélange et répartition en un set de test et un set d'apprentissage
data = getMushroom()
random.shuffle(data)
cut = len(data)/3
test_list = data[0:cut]
learn_list = data[cut:len(data)]

# Lancement de l'algorithme de perceptron renvoyant w0 et w.
T = 10 * cut
eta = 1
epsilon = 1
print "--- Paramètres ---"
print "T : " + str(T)
print "eta : " + str(eta)
print "epsilon : " + str(epsilon)

print "--- Performances ---"
print "PERCEPTRON"
wT = perceptron(learn_list,T,eta)
perf = testMethode(test_list,wT)
print perf
choixEta(data,5,T)

print "ADALINE"
wT = adaline(learn_list, T, eta, epsilon)
perf = testMethode(test_list,wT)
print perf
print wT

print "LOGISTIQUE"
wT = logistique(learn_list, T, eta, epsilon)
perf = testMethode(test_list,wT)
print perf
print wT

print "EXPONENTIELLE"
wT = exponentielle(learn_list, T, eta, epsilon)
perf = testMethode(test_list,wT)
print perf

# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
