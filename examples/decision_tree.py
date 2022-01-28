import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

df = pandas.read_csv("show.csv")

print(df)
print(type(df))
# df.plot() Hier wid ds ganze als Plot angezeigt

renamenationalitydic = {'UK': 0, 'USA': 1, 'DE': 2}
df['Nationality'] = df['Nationality'].map(renamenationalitydic)
renamegodic = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(renamegodic)
print()
print(df)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print()
print()
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree)

export = export_text(dtree, feature_names=features)
print(export)

feedback = dtree.predict([[40, 10, 7, 1]])

if feedback == 1:
    feedback = 'Yes'
else:
    feedback = 'No'
print(feedback)
