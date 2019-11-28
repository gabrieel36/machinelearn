from sklearn.datasets import load_iris
iris = load_iris()

#1º algoritmo
#importando o classificador
from sklearn.linear_model import SGDClassifier
#ndefinindo o numero de iterações = 50;
clas = SGDClassifier(loss="hinge", penalty="l2", max_iter=50)
#treinamento
tr1 = clas.fit(iris.data, iris.target)
#acurácia
x1=tr1.score(iris.data, iris.target)

#2º algoritmo
#importando o classificador
from sklearn.neighbors import KNeighborsClassifier
clas2 = KNeighborsClassifier(n_neighbors=3)
#treinamento
tr2 = clas2.fit(iris.data, iris.target)
#acurácia
x2=tr2.score(iris.data, iris.target)

#3º algoritmo
#importando o classificador
from sklearn.ensemble import RandomForestClassifier
clas3 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#treinamento
tr3 = clas3.fit(iris.data, iris.target)
#acurácia
x3=tr3.score(iris.data, iris.target)


#4º algoritmo
#importando o classificador
from sklearn.naive_bayes import GaussianNB
clas4 = GaussianNB()
#treinamento
tr4 = clas4.fit(iris.data, iris.target)
#acurácia
x4=tr4.score(iris.data, iris.target)

#exibindo a acurácia:
print("Accuracy SGD: ", tr1.score(iris.data, iris.target))
print("Accuracy KNeighborsClassifier: ", tr2.score(iris.data, iris.target))
print("Accuracy RandomForest: ", tr3.score(iris.data, iris.target))
print("Accuracy Naive Bayes: ", tr4.score(iris.data, iris.target))

#grafico
import matplotlib.pyplot as plt
import numpy as np

pop = {"Stochastic Gradient Descent (SGD)": x1*100, "Neighbors": x2*100, "RandomForest": x3*100, "Naive Bayes": x4*100}
alg = [i for i in pop.keys()]
valor = [j for j in pop.values()]
aux2 = np.arange(len(alg))


plt.barh(aux2, valor, align='center', color="green")
plt.yticks(aux2, alg)

plt.title('Acurácia %')







