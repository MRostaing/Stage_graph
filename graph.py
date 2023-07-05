from random import random
from copy import deepcopy
from string import ascii_lowercase
import matplotlib.pyplot as plt


def est_valide(matrice):
    """
    entrée: matrice sous forme "classique"
    sortie: bouléen
    fonction: détermine si la matrice en entrée respecte les conditions
    """
    for i in range(1, len(matrice) - 1):
        if len(matrice[i - 1]) - 1 != len(matrice[i]):
            return False

    for i in range(len(matrice)):
        if matrice[i][len(matrice[i]) - 1] != i + 1:
            return False

    for i in range(len(matrice)):
        for j in range(len(matrice[i]) - 1):
            if (
                matrice[i][j] != matrice[i][j + 1]
                and matrice[i][j] != matrice[matrice[i][j + 1]][j]
            ):
                return False

    return True


def desensemblifie(mat):
    """
    Entrée: matrice sous forme "ensemble"
    Sortie: matrice sous forme "classique"
    Fonction: transphorme une matrice en notation ensemble en une matrice en notation classique
    """
    for i in range(len(mat)):
        for j in range(len(mat) - i):
            mat[i][j].discard(min(mat[i][j]))
            mat[i][j] = min(mat[i][j])
    return mat


def succeseur(tab):
    """
    Entrée: une liste de 0 et de 1
    Sortie: une liste de 0 et de 1
    Fonction: transphorme le nombre représenter en binaire par la liste en le nombre suivant
    """
    i = len(tab) - 1
    while (i >= 0) and (tab[i] == 1):
        tab[i] = 0
        i -= 1

    if i >= 0:
        tab[i] = 1
    return tab


def tout_les_nombres(n):
    """
    Entrée: int
    Sortie: liste de liste
    Fonction: donner la liste des nombre binaire contenant n bit
    """
    nombres = []
    no = []
    for i in range(n):
        no.append(0)

    for i in range(2**n):
        nombres.append(no.copy())
        no = succeseur(no)
    return nombres


def genere_suiv(base, tab):
    """
    Entrée: matrice sous forme classique, liste de 0 et de 1
    Sortie: matrice sous la forme classique
    Fonction: génère une matrice de taille n+1 à partir de la matrice de base et de la liste
    """
    for i in range(len(base)):
        base[i] = [0] + base[i]
    base += [[len(base) + 1]]
    for i in range(len(tab) - 1, -1, -1):
        if tab[i] == 0:
            base[i][0] = base[i][1]
        if tab[i] == 1:
            base[i][0] = base[base[i][1]][0]
    return base


def retourne_nombre(ensemble: set):
    """
    Entrée: set
    Sortie: int
    Fonction: rend le nombre correspondant à la notation classique à partir de la notation en ensemble
    """
    new = ensemble.copy()
    new.discard(min(new))
    return min(new)


def sigma(base, key):
    """
    Entrée: matrice en notation classique, liste de 0 et 1
    Sortie: liste
    Fonction: rend la permutation associée au chemin
    """
    mat = deepcopy(base)
    sigma = []
    i = 0
    j = len(mat) - 1
    possiblities = set()
    for k in range(len(mat) + 1):
        possiblities.add(k)
    for k in range(len(key) - 1):
        if key[k]:
            # print(mat[i][j])
            sigma.append((mat[i][j] - mat[i][j - 1]).pop())
            j -= 1
        if not (key[k]):
            sigma.append((mat[i][j] - mat[retourne_nombre(mat[i][j])][j - 1]).pop())
            i = retourne_nombre(mat[i][j])
            j -= 1
    if key[-1]:
        sigma.append(max(mat[i][j]))
    if not key[-1]:
        sigma.append(min(mat[i][j]))
    sigma.append((possiblities - set(sigma)).pop())
    return sigma


def bucket_matrice(matrice):
    """
    entrée: matrice notation ensemble
    sortie: dictionnaire
    Fonction: rend un dictionnaire qui répertorie les ensemble en fonction de leur taille
    """
    base = deepcopy(matrice)
    dico = dict()
    for i in range(len(base)):
        dico[i] = []
    for i in range(len(matrice)):
        for j in range(len(matrice[i])):
            dico[len(matrice[i][j]) - 2].append(matrice[i][j])
    return dico


def traite_permutation(mat, sig):
    """
    Entrée: matrice notation ensemble, liste corespondant à la permutation
    Sortie: matrice
    Fonction: Réaffecte les indice en fonction de la permutation
    """
    new = set()
    matrice = deepcopy(mat)
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            for elt in mat[i][j]:
                for k in range(len(sig)):
                    if elt == sig[k]:
                        new.add(k)
            matrice[i][j] = new
            new = set()

    return matrice


def genere_ensemble(mat):
    """
    Entrée: matrice en notation classique
    Sortie: matrice en notation ensemble
    Fonction: transforme la matrice en notation ensemble"""
    n = len(mat)
    mat_ensemble = []
    for i in range(n):
        mat_ensemble.append([0])
        for j in range(n - i - 1):
            mat_ensemble[i] += [0]

    for i in range(n - 1, -1, -1):
        mat_ensemble[i][0] = set([mat[i][0], i])

    for j in range(1, n):
        for i in range(n - 1 - j, -1, -1):
            mat_ensemble[i][j] = set([i]).union(mat_ensemble[mat[i][j]][j - 1])
    return mat_ensemble


def bouge_label(label, sigma):
    new_label = []
    for i in range(len(sigma)):
        new_label.append(label[sigma[i]])
    return new_label


def permutation(matrice, key):
    """
    Entrée: matrice en notation classique, liste de 0, 1
    Sortie: matrice
    Fonction: rend la permutation de la matrice"""
    sig = sigma(matrice, key)
    dico = bucket_matrice(matrice)
    mat = deepcopy(matrice)
    a = 0
    b = len(matrice) - 1
    for k in range(len(sig) - 1):
        for i in range(len(dico) - 1, -1, -1):
            for j in range(len(dico[i]) - 1, -1, -1):
                if sig[k] in dico[i][j]:
                    mat[a][b] = dico[i][j]
                    dico[i].pop(j)
                    b -= 1
                    if b < 0:
                        b = len(matrice[a]) - 2
                        a += 1
    return traite_permutation(mat, sig)


def rajoute_pied(mat):
    """
    Entrée: matrice en notation classique
    Sortie: matrice en notation classique
    Fonction: rajoute un pied à la matrice"""
    for i in range(len(mat)):
        mat[i].append(i + 1)
    mat.append([len(mat) + 1])
    return mat


def matrice_binaire(mat):
    """
    Entrée: matrice en notation classique
    Sortie: matrice en notation binaire
    Fonction: transforme la matrice en notation binaire"""
    tab = []
    n = len(mat)
    for i in range(n - 1):
        tab.append([1])
        for j in range(n - i - 2):
            tab[i] += [1]
    # print(tab)
    for j in range(n - 2, -1, -1):
        for i in range(n - j - 1):
            if mat[i][j] == mat[i][j + 1]:
                tab[j][i] = 0

    return tab


def binaire_matrice(mat):
    """
    Entrée: matrice en notation binaire
    Sortie: matrice en notation classique
    Fonction: transforme la matrice en notation classique"""
    tab = [[1]]
    n = len(mat)
    for i in range(n - 1, -1, -1):
        tab = genere_suiv(tab, mat[i])
    return tab


class Graph:
    def __init__(self, mat):
        try:
            self.mat = []
            self.label = []
            for i in range(mat):
                if i < 26:
                    self.label.append(ascii_lowercase[i])
                self.mat.append([i + 1])
                for j in range(mat - i - 1):
                    self.mat[i] += [i + 1]
        except:
            self.mat = mat
            self.label = []
            for i in range(len(mat) + 1):
                if i < 26:
                    self.label.append(ascii_lowercase[i])

    def genere_aleatoire(self, p=0.5):
        for i in range(len(self.mat)):
            self.mat[i][len(self.mat[i]) - 1] = i + 1

        for j in range(len(self.mat) - 2, -1, -1):
            for i in range(len(self.mat) - 2 - j, -1, -1):
                # print(i, j)
                coin = random()
                if coin <= p:
                    # print(self.mat[i][j + 1])
                    self.mat[i][j] = self.mat[i][j + 1]

                if coin > p:
                    # print("b")
                    self.mat[i][j] = self.mat[self.mat[i][j + 1]][j]
        return self.mat

    def genere_ensemble(self):
        n = len(self.mat)
        mat_ensemble = []
        for i in range(n):
            mat_ensemble.append([0])
            for j in range(n - i - 1):
                mat_ensemble[i] += [0]

        for i in range(n - 1, -1, -1):
            mat_ensemble[i][0] = set([self.mat[i][0], i])

        for j in range(1, n):
            for i in range(n - 1 - j, -1, -1):
                mat_ensemble[i][j] = set([i]).union(mat_ensemble[self.mat[i][j]][j - 1])
        return mat_ensemble

    def genere_suiv(self, tab):
        for i in range(len(self.mat)):
            self.mat[i] = [0] + self.mat[i]
        self.mat += [[len(self.mat) + 1]]
        for i in range(len(tab) - 1, -1, -1):
            if tab[i] == 0:
                self.mat[i][0] = self.mat[i][1]
            if tab[i] == 1:
                self.mat[i][0] = self.mat[self.mat[i][1]][0]
        return self.mat

    def __str__(self):
        x = ""
        for i in range(len(self.mat)):
            x += str(self.label[i]) + ":" + str(self.mat[i]) + "\n"
        return x

    def genere_tout(self):
        dico = {1: [[[1]]]}
        tab = []
        for i in range(2, len(self.mat) + 1):
            dico[i] = []
        for i in range(1, len(self.mat)):
            tab = tout_les_nombres(i)
            for j in range(len(tab)):
                for k in range(len(dico[i])):
                    dico[i + 1].append(genere_suiv(deepcopy(dico[i][k]), tab[j]))
        return dico

    def genere_permutations(self):
        ensemble = set()
        tab = tout_les_nombres(len(self.mat))
        mat = genere_ensemble(self.mat)
        for i in range(len(tab)):
            new = deepcopy(permutation(mat, tab[i]))
            new = desensemblifie(new)
            new = tuple(tuple(i) for i in new)
            ensemble.add(new)
        return ensemble

    def feet(self):
        key = [1 for i in range(len(self.mat))]
        mat = genere_ensemble(self.mat)
        sig = sigma(mat, key)
        self.label = bouge_label(self.label, sig)
        return [0, sig[0]], self.label

    def classes(self):
        graph = Graph(self.mat)
        tab = graph.genere_tout()[len(self.mat)]
        pied_par_terre = []
        classe = set()
        for i in range(len(tab)):
            mat = genere_ensemble(tab[i])
            for k in range(3):
                if k < 1:
                    pied_par_terre.append(
                        desensemblifie(permutation(mat, [1 for j in range(len(mat))]))
                    )
                else:
                    pied_par_terre.append(
                        desensemblifie(
                            permutation(
                                genere_ensemble(pied_par_terre[-1]),
                                [1 for j in range(len(mat))],
                            )
                        )
                    )

            sous_classe = set([tuple(tuple(a) for a in pied_par_terre[1])])
            sous_classe.add(tuple(tuple(b) for b in pied_par_terre[2]))
            pied_par_terre = []
            classe.add(frozenset(sous_classe))
        return classe

    def pourcentage_symetrique(self):
        A = Graph(self.mat)
        classe = A.class_opti()
        total = len(classe)
        symetrique = 0
        for i in classe:
            if len(i) == 1:
                symetrique += 1
        return "il y a ", ((symetrique / total) * 100), "%" "de matrice symétrique"

    def donne_symetrique(self):
        tab = []
        A = Graph(self.mat)
        classe = A.class_opti()
        for i in classe:
            if len(i) == 1:
                tab.append(set(i).pop())
        return tab

    def class_opti(self):
        rg = len(self.mat) - 1
        graph = Graph(rg)
        tab = graph.genere_tout()[rg]
        classes = set()
        for i in range(len(tab)):
            pied_sol = [rajoute_pied(tab[i])]
            mat = genere_ensemble(pied_sol[0])
            pied_sol.append(
                desensemblifie(permutation(mat, [1 for k in range(len(mat))]))
            )
            sous_classe = set([tuple(tuple(a) for a in pied_sol[0])])
            sous_classe.add(tuple(tuple(b) for b in pied_sol[1]))
            pied_sol = []
            classes.add(frozenset(sous_classe))
        return classes

    def classe(self):
        pied_par_terre = []
        classe = set()
        mat = genere_ensemble(self.mat)
        print(mat)
        for k in range(3):
            if k < 1:
                pied_par_terre.append(
                    desensemblifie(permutation(mat, [1 for j in range(len(mat))]))
                )
                print(pied_par_terre)

            else:
                pied_par_terre.append(
                    desensemblifie(
                        permutation(
                            genere_ensemble(pied_par_terre[-1]),
                            [1 for j in range(len(mat))],
                        )
                    )
                )

        sous_classe = set([tuple(tuple(a) for a in pied_par_terre[1])])
        sous_classe.add(tuple(tuple(b) for b in pied_par_terre[2]))
        classe.add(frozenset(sous_classe))
        return classe

    def represente(self, mat=None):
        if mat == None:
            mat = self.mat
        x = []
        y = []
        dico = {}
        for i in range(len(mat) + 1):
            dico[i] = []

        for i in range(len(mat) + 1):
            try:
                for j in range(len(mat[i]) + 1):
                    x.append(j)
                    y.append(-i)
                    dico[i].append([j, -i])
            except:
                x.append(x[0])
                y.append(y[-1] - 1)
                dico[i].append([x[0], y[-1]])

        fig, ax = plt.subplots(figsize=(15, 15))

        ax.axis("equal")

        ax.set_title("graphique")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.plot(x, y, marker="x", linestyle="None")

        for i in range(len(dico)):
            ax.plot(
                [dico[i][0][0], dico[i][-1][0]],
                [dico[i][0][1], dico[i][0][1]],
                color="red",
            )

        for i in range(len(dico)):
            ax.plot(
                [i, i],
                [0, -len(dico) + 1],
                color="green",
                linestyle="--",
            )

        for i in range(len(mat) - 1, -1, -1):
            for j in range(len(mat[i]) - 1, -1, -1):
                ax.plot(
                    [dico[i][j + 1][0], dico[i][j][0]],
                    [dico[i][j][1], dico[mat[i][j]][j][1]],
                    color="blue",
                )

        plt.show()
