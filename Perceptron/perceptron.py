#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np

# Implémentation de l'algorithme de Perceptron pour classifier des données en deux classes

# Tout d'abord nous allons modifier le fichier iris.data de sorte que les iris-setosa appartiennent à la classe +1
# et que les autres à la classe -1

# S la base d'apprentissage
# T le nombre max d'itérations
# eta le pas d'apprentissage
def perceptron (S, T, eta) :
    size_of_space = len(S[0]) -1
    w0 = [0]
    w = [ [0] * size_of_space ]
    for t = 1..T :
        for i = 1..len(S) :
            yi = S[i][4]
            if  yi * (w0[t] + <w[t],xi>) <= 0 :
                w0[t+1] <- w0[t] + eta * yi
                w[t+1] <- w[t] + eta * xi *yi
    print w0

# BT la base de test
# w0
# w
def testPerceptron(BT, w0, w) :
    print hi

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
    return lines


# Récupération des données, mélange et répartition en un set de test et un set d'apprentissage
data = getIris()
random.shuffle(data)
cut = len(data)/3
test_list = data[0:cut]
learn_list = data[cut:len(data)]

# Lancement de l'algorithme de perceptron renvoyant w0 et w.
perceptron(learn_list,100,0.1)


# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
