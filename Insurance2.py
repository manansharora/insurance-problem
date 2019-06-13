import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('D:/Coding/data/train.csv')
test_data = pd.read_csv('D:/Coding/data/test.csv')

y = train_data[['Claim']].copy()

details = ['ID', 'Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Duration',
            'Destination', 'Net Sales', 'Commision (in value)',
                'Gender', 'Age']

x = train_data[details].copy(deep = True)

claim_checker = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)

claim_checker.fit(x,y)

predictions = claim_checker.predict(test_data)

print(predictions[:10])

precision = precision_score()
accuracy = accuracy_score()

print(precision)
print(accuracy)
