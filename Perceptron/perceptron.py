#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np

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

# S la base d'apprentissage
# T le nombre max d'itérations
# eta le pas d'apprentissage
def perceptron (S, T, eta) :
    size_of_space = len(S[0]) -1
    w0 = np.zeros((T+1,1))
    w = np.zeros((T+1,size_of_space))
    for t in range(0,T) :
        for i in range(0,len(S)) :
            yi = S[i][size_of_space]
            if  yi * (w0[t] + np.dot(w[t], S[i][0:size_of_space])) <= 0 :
                w0[t+1] = w0[t] + eta * yi
                w[t+1] = w[t] + eta * S[i][0:size_of_space] * yi
    return w[T]

# BT la base de test
# wT le résultat du perceptron
def testPerceptron(BT, wT) :
    size_of_space = len(BT[0]) -1
    perf = 0
    for i in range(0,len(BT)) :
        yi = BT[i][size_of_space]
        if yi * np.dot(wT, BT[i][0:size_of_space]) :
            perf += 1
    return perf/len(BT)

# S le set de données
# k le nombre de sous-ensembles sur lesquels faire la cross-validation
def choixEta(S,k) :
    size = len(S)
    eta_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    res = []
    for eta in eta_range :
        for i in k :
        print "Résultat pour eta = " + eta



# Récupération des données, mélange et répartition en un set de test et un set d'apprentissage
data = getIris()
random.shuffle(data)
cut = len(data)/3
test_list = data[0:cut]
learn_list = data[cut:len(data)]

# Lancement de l'algorithme de perceptron renvoyant w0 et w.
res = perceptron(learn_list,100,0.1)
perf = testPerceptron(test_list,res)
print res
print perf
choixEta(data,4)

# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
