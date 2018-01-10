#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from operator import add

# Implémentation de l'algorithme de Perceptron pour classifier des données en deux classes

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
    return np.array(lines, dtype=float) # Convertir le contenu du tableau en float


def normalize(a) :
    size = a.shape[1]
    maxs = a.max(axis=0)
    np.append(maxs,[1])
    print maxs
    normalized = a/maxs
    return normalized

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
        res = map(lambda x : (ord(x) - 96), res)
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

# BT la base de test
# wT le résultat du perceptron
def testPerceptron(BT, wT) :
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
            res[i] += testPerceptron(test_list,wT)
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
data = getIris()
random.shuffle(data)
cut = len(data)/3
test_list = data[0:cut]
learn_list = data[cut:len(data)]

# Lancement de l'algorithme de perceptron renvoyant w0 et w.
T = 10 * cut
wT = perceptron(learn_list,T,0.1)
perf = testPerceptron(test_list,wT)
print wT
print perf
choixEta(data,10,T)

# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
