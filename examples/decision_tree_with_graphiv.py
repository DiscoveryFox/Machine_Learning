import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("show.csv")

renamenationalitydic = {'UK': 0, 'USA': 1, 'DE': 2}
df['Nationality'] = df['Nationality'].map(renamenationalitydic)
renamegodic = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(renamegodic)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
print('X: ')
print(X)
y = df['Go']

dtree = DecisionTreeClassifier( )
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img = pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show( )
