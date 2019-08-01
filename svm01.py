import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import glob, os.path, re, json
import matplotlib.pyplot as plt

mr = pd.read_csv("mushroom.csv", header = None)

data = []
label = []
attr_list = []

for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    exdata = []

    for col, v in enumerate(row.ix[1:]):
        if row_index == 0:
            attr = {"dic":{}, "cnt":0}
            attr_list.append(attr)

        else:
            attr = attr_list[col]

        d = [0,0,0,0,0,0,0,0,0,0,0,0]

        if v in attr["dic"]:
            idx = attr["dic"][v]

        else:
            idx =attr["cnt"]
            attr["dic"][v] = idx
            attr["cnt"] += 1

        d[idx] = 1
        exdata += d
    data.append(exdata)

""" for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    row_data = []
    for v in row.ix[1:]:
        row_data.append(ord(v))
    data.append(row_data) """

print(label[:5])
print("============================")
print(data[:5])
""" 
data_train, data_test, label_train, label_test = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)

print("SCORE =", ac_score)
print("REPORT =", cl_report) """




""" 
files = glob.glob("./test/*.txt")

test_data = []
test_label = []
for file_name in files:
    basename = os.path.basename(file_name)
    lang = basename.split("-")[0]
    file = open(file_name, "r", encodoing = "utf-8")
    text = file.read()
    text = text.lower()
    file.close()
    
    code_a = ord("a")
    code_z = ord("z")
    count = [0 for n in range(0, 26) ]
    for character in text:
        code_current = ord(character)
        if code_a <= code_current <= code_z:
            count[code_current - code_a] += 1
    
    total = sum(count)
    count = list(map(lambda n: n/total, count))
    test_data.append(count)
    test_label.append(lang)


    clf = svm.SVC()
    clf.fit(train_data, train_label)
    predict = clf.predict(test_data)
    score = metrics.accuracy_score(test_label, predict)
    print("score=", score)
    report = metrics.classification_report(test_label, predict) """


