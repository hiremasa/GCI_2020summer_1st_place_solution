# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# In[2]:
#pathは各自適切に指定してください
path = "/content/drive/My Drive/Kaggle_titanic/"
train_df = pd.read_csv(path+"train.csv")
test_df = pd.read_csv(path+"test.csv")
data_df = train_df.append(test_df)

#Pseudo Labeling
# In[3]:前処理
def process_names(df):
    df["FamilyName"]=df["Name"].map(lambda x: x.split(",")[0].strip())
    df["FullName"]=df["Name"].map(lambda x: x.split(",")[1].strip())
    df["Title"]=df["FullName"].map(lambda x: x.split(".")[0].strip())
    df["TwoLetters"]=df["FamilyName"].map(lambda x: x[-2:])

process_names(train_df)
process_names(test_df)

train_df["TicketButLast"]=train_df.Ticket.map(lambda x: x[:-1])
test_df["TicketButLast"]=test_df.Ticket.map(lambda x: x[:-1])

ticket_count_train = train_df["Ticket"].value_counts()
ticket_count_test = test_df["Ticket"].value_counts()
ticket_inter = np.intersect1d(train_df["Ticket"].values,test_df["Ticket"].values)

ticketButLast_count_train = train_df["TicketButLast"].value_counts()
ticketButLast_count_test = test_df["TicketButLast"].value_counts()
ticketButLast_inter = np.intersect1d(train_df["TicketButLast"].values,test_df["TicketButLast"].values)

for idx in train_df.index:
    ticket = train_df.loc[idx,"Ticket"]
    train_df.loc[idx,"CountTicket_InTrain"]=ticket_count_train[ticket]
    if(ticket in ticket_inter):
        train_df.loc[idx,"CountTicket"]=(ticket_count_train[ticket]+ticket_count_test[ticket])
    else:
        train_df.loc[idx,"CountTicket"]=ticket_count_train[ticket]

for idx in test_df.index:
    ticket = test_df.loc[idx,"Ticket"]
    test_df.loc[idx,"CountTicket_InTest"]=ticket_count_test[ticket]
    if(ticket in ticket_inter):
        test_df.loc[idx,"CountTicket"]=(ticket_count_train[ticket]+ticket_count_test[ticket])
    else:
        test_df.loc[idx,"CountTicket"]=ticket_count_test[ticket]

for idx in train_df.index:
    ticketButLast = train_df.loc[idx,"TicketButLast"]
    train_df.loc[idx,"CountTicketButLast_InTrain"]=ticketButLast_count_train[ticketButLast]
    if(ticketButLast in ticketButLast_inter):
        train_df.loc[idx,"CountTicketButLast"]=(ticketButLast_count_train[ticketButLast]
                                               +ticketButLast_count_test[ticketButLast])
    else:
        train_df.loc[idx,"CountTicketButLast"]=ticketButLast_count_train[ticketButLast]

for idx in test_df.index:
    ticketButLast = test_df.loc[idx,"TicketButLast"]
    test_df.loc[idx,"CountTicketButLast_InTest"]=ticketButLast_count_test[ticketButLast]
    if(ticketButLast in ticketButLast_inter):
        test_df.loc[idx,"CountTicketButLast"]=(ticketButLast_count_train[ticketButLast]
                                               +ticketButLast_count_test[ticketButLast])
        test_df.loc[idx,"CountTicketButLast_InTrain"]=(ticketButLast_count_train[ticketButLast])
    else:
        test_df.loc[idx,"CountTicketButLast"]=ticketButLast_count_test[ticketButLast]

for idx in train_df.index:
    ticket = train_df.loc[idx,"Ticket"]
    if(ticket in ticket_inter):
        train_df.loc[idx,"FareCorrect"]=train_df.loc[idx,"Fare"]/(ticket_count_train[ticket]
                                                                +ticket_count_test[ticket])
    else:
        train_df.loc[idx,"FareCorrect"]=train_df.loc[idx,"Fare"]/(ticket_count_train[ticket])
#train_df.head()

# In[4]:Pseudo labelingの対象となるtestデータのindexを取得する
#下のカーネルの分析を参照
#https://www.kaggle.com/hirehire2/titanic-top-score-made-easy
index1=test_df[test_df.Ticket=="3101295"].index #0と予想できる　上記Kernel In[48]あたりを参照

index2=test_df[test_df.FamilyName=="Peacock"].index #0だと予想できる　上記Kernel In[46]あたりを参照

index3=test_df[(test_df.CountTicketButLast>10)&(test_df.Pclass==3)&(test_df.SibSp==0)&(test_df.Parch==0)&(test_df.Sex=="female")&(test_df.Title=="Miss")].index#0だと予想できる　上記Kernel In[57]あたりを参照

index4=test_df[(test_df.TwoLetters.isin(["ic","ff"]))&(test_df.Sex=="female")].index #0だと予想できる　上記Kernel In[61]あたりを参照

index5=test_df[test_df.TicketButLast=="34708"].index#0だと予想できる　上記Kernel In[59]あたりを参照

dfFamilyTrain=train_df[(train_df["Parch"]>0)&(train_df["Survived"]==0)]
dfFamily=dfFamilyTrain[(dfFamilyTrain["Sex"]=="female")|(dfFamilyTrain["Age"]<10)]
familiestrain=dfFamily["FamilyName"]
dfFamilytest=test_df[(test_df["Parch"]>0)&(test_df["Sex"]=="female")]
familiestest=dfFamilytest["FamilyName"]
intersection=np.intersect1d(familiestest,familiestrain)
intersection #array(['Abbott', 'Aks', 'Asplund', 'Drew', 'Peter', 'Ryerson', 'Spedden',Touma', 'Wells'], dtype=object)
index6=dfFamilytest[dfFamilytest["FamilyName"].isin(intersection)]["PassengerId"].index #0だと予想できる 上記Kernel In[38]あたりを参照

index7=test_df[test_df.Ticket=="1601"].index #1だと予想できる　上記Kernel In[50]あたりを参照

# In[5]: trainデータに追加する
del train_df, test_df, dfFamilyTrain, dfFamily, dfFamilytest
train_df = pd.read_csv(path+"train.csv")
test_df = pd.read_csv(path+"test.csv")
pseudo_df=pd.concat([test_df.iloc[index1], test_df.iloc[index2],  test_df.iloc[index3],  test_df.iloc[index4],  test_df.iloc[index5], test_df.iloc[index6], test_df.iloc[index7]],axis=0)
pseudo_df["Survived"]=0
pseudo_df.loc[index7, "Survived"]=train_df[train_df.Ticket=="1601"].Survived.mode()[0] #1だと予想できるが0も混ざっているためtrainデータの最頻値にした
#pseudo_df
train_df=pd.concat([train_df, pseudo_df], axis=0)
train_df.reset_index(drop=True,inplace=True)
print(train_df.shape) #(910, 12)
data_df = train_df.append(test_df)

# In[6]: kNN Feature Engineering
#下のKernelと同じなので参考にして下さい
#https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83
len_train=len(train_df)
data_df['Title'] = data_df['Name']
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    train_df['Age'] = data_df['Age'][:len_train]
test_df['Age'] = data_df['Age'][len_train:]
data_df.drop('Title', axis = 1, inplace = True)

data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
train_df['Family_Size'] = data_df['Family_Size'][:len_train]
test_df['Family_Size'] = data_df['Family_Size'][len_train:]

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
#data_df['Last_Name'] =data_df['Name'].apply(lambda x: str.split(x, ".")[1].split()[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0


for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0


train_df['Family_Survival'] = data_df['Family_Survival'][:len_train]
test_df['Family_Survival'] = data_df['Family_Survival'][len_train:]

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:len_train]
test_df['FareBin_Code'] = data_df['FareBin_Code'][len_train:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)

data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:len_train]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][len_train:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)

# In[7]: training
X_train = train_df.drop('Survived', 1)
y_train = train_df['Survived']
X_test = test_df.copy()

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

n_neighbors = [8, 10, 12, 14, 16]
hyperparams = {'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(leaf_size=10), param_grid = hyperparams, verbose=True,
                cv=4, scoring = "f1")
gd.fit(X_train, y_train)
#print(gd.best_estimator_)
#KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=14, p=2,
#                     weights='uniform')
gd.best_estimator_.fit(X_train, y_train)

# In[7]: submit
y_pred=gd.best_estimator_.predict(X_test)
submission=pd.read_csv(path+"submission.csv")
output = pd.DataFrame({'PassengerId': submission.PassengerId, 'Survived': y_pred})
output.to_csv(path+'GCI_submission.csv', index=False)
