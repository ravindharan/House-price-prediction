#logistic regression model import
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#reading training set file
tr=open("train_after_extracting_Number_Columns.csv","r")
records=tr.readlines()
tr.close()

#Making training set X and y vectors
X=[[] for i in range(1460)]
y=[]
for i in range(1,len(records)):
    for j in range(len(records[i].strip().split(","))-1):
        X[i - 1].append(int(records[i].strip().split(",")[j]))
    y.append(int(records[i].strip().split(",")[36]))

#training our logistic regression model
lr = LogisticRegression()
lr.fit(X,y)

#reading testing set file
te=open("test_after_extracting_Number_Columns.csv","r")
records1=te.readlines()
te.close()

#Making testing set X vector
XX=[[] for i in range(1459)]
yy=[]
for i in range(1,len(records1)):
    for j in range(len(records1[i].strip().split(","))):
        XX[i - 1].append(int(records1[i].strip().split(",")[j]))

yy = lr.predict(XX)

# writing predicted house price to new file
result=open("result.csv","w")
result.write("House No,Predicted Price" + "\n")
for i in range(len(yy)):
    result.write(str(i+1) + "," + str(yy[i]) + "\n")
result.close()

#Checking for model accuracy by applying model on training set
yyy = lr.predict(X)
print ("model accuracy: {}%").format(accuracy_score(y,yyy)*100)



