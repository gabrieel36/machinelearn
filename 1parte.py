print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
x1=97
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


#2º algoritmo
#importando o classificador
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(random_state=0)
#treinamento
tr = clf.fit(digits.data, digits.target)
#acurácia
x2=tr.score(digits.data, digits.target)
#grafico decisionTree
plt.show(tree.plot_tree(clf.fit(digits.data, digits.target)))

#3º algoritmo
#importando o classificador
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#treinamento
tr1 = neigh.fit(digits.data, digits.target)
#acurácia
x3 = tr1.score(digits.data, digits.target)

#exibindo a acurácia:
print("Accuracy Neighbors:", tr1.score(digits.data, digits.target))
print("Accuracy DecisionTree: ", tr.score(digits.data, digits.target))
print("Accuracy SVM: ", x1)


#criando o gráfico
import matplotlib.pyplot as plt
import numpy as np

pop = {"SVM": x1, "Decision Tree": x2*100, "KNeighborsClassifier": x3*100}
alg = [i for i in pop.keys()]
valor = [j for j in pop.values()]
aux2 = np.arange(len(alg))
plt.barh(aux2, valor, align='center', color="blue")
plt.yticks(aux2, alg)
plt.title('Acurácia %')

plt.show()








