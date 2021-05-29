from sklearn import datasets
from model_selection.kfold import KFold

iris = datasets.load_iris()
X = iris.data
y = iris.target

kf = KFold(5)

for train, test in kf.split(X, y):
    print(train, test)

