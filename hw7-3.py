import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

titanic_df = pd.read_csv("train (1).csv")


titanic_df["Sex"].replace(to_replace="male",value=1,inplace=True)
titanic_df["Sex"].replace(to_replace="female",value=0,inplace=True)

dummie= pd.get_dummies(titanic_df["Embarked"])
titanic_df = pd.concat([titanic_df,dummie],axis=1)
titanic_df.drop(["Embarked"], inplace=True, axis=1)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df["C"]=titanic_df["C"].astype(np.int64)
titanic_df["Q"]=titanic_df["Q"].astype(np.int64)
titanic_df["S"]=titanic_df["S"].astype(np.int64)
titanic_df["Fare"]=titanic_df["Fare"].astype(np.int64)
titanic_df["Age"]=titanic_df["Age"].astype(int)

X = titanic_df[["Pclass","Sex","Age","SibSp","Parch","Fare","C","Q","S"]]
Y = titanic_df["Survived"]

log_reg = LogisticRegression()

X_eğitim, X_test, Y_eğitim, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=111)

log_reg.fit(X_eğitim, Y_eğitim)

egitim_dogruluk = log_reg.score(X_eğitim, Y_eğitim)
test_dogruluk = log_reg.score(X_test, Y_test)

print('One-vs-rest', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')
print("\n\n")

tahmin_eğitim = log_reg.predict(X_eğitim)
tahmin_test = log_reg.predict(X_test)
tahmin_test_ihtimal = log_reg.predict_proba(X_test)[:,1]

hata_matrisi_eğitim = confusion_matrix(Y_eğitim, tahmin_eğitim)
hata_matrisi_test = confusion_matrix(Y_test, tahmin_test)

print("Hata Matrisi (Eğitim verileri)", "-"*30, hata_matrisi_eğitim, sep="\n")
print()
print("Hata Matrisi (Test verileri)", "-"*30, hata_matrisi_test, sep="\n")
print()

TN = hata_matrisi_test[0][0]
TP = hata_matrisi_test[1][1]
FP = hata_matrisi_test[0][1]
FN = hata_matrisi_test[1][0]

print("Doğru negatif sayısı   :", TN)
print("Doğru pozitif sayısı   :", TP)
print("Yanlış pozitif sayısı  :", FP)
print("Yanlış negatif sayısı  :", FN)

print("\nDoğruluk değerleri")
print("Modelden alınan doğruluk değeri : ",  log_reg.score(X_test, Y_test))
print("Hesaplanan doğruluk değeri      : ",  (TN + TP)/(FN + FP + TN + TP))
print("accuracy_score() değeri         : ",  accuracy_score(Y_test, tahmin_test))

print("\nHata oranı: ", 1-accuracy_score(Y_test, tahmin_test))

print("\nHassasiyet oranı: ")
print("Hesaplanan doğruluk değeri      : ",  (TP)/(FP + TP))
print("precision_score() değeri        : ",  precision_score(Y_test, tahmin_test))

print("\nDuyarlılık oranı: ")
print("Hesaplanan doğruluk değeri   : ",  (TP)/(TP + FN))
print("recall_score() değeri        : ",  recall_score(Y_test, tahmin_test))

print("\nHesaplanan özgünlük değeri   : ",  (TN)/(TN + FP))

hassasiyet_degeri = precision_score(Y_test, tahmin_test)
duyarlılık_değeri = recall_score(Y_test, tahmin_test)


print("Hesaplanan f1 skoru   : ",  2*((hassasiyet_degeri*duyarlılık_değeri)/(hassasiyet_degeri + duyarlılık_değeri)))
print("f1_score() değeri     : ",  f1_score(Y_test, tahmin_test))

print(classification_report(Y_test,tahmin_test) )

print("f1_score() değeri        : {:.2f}".format(f1_score(Y_test, tahmin_test)))
print("recall_score() değeri    : {:.2f}".format(recall_score(Y_test, tahmin_test)))
print("precision_score() değeri : {:.2f}".format(precision_score(Y_test, tahmin_test)))
print('\n')

metrikler =  precision_recall_fscore_support(Y_test, tahmin_test)
print("Hassasiyet :" , metrikler[0])
print("Duyarlılık :" , metrikler[1])
print("F1 Skoru   :" , metrikler[2])


print('AUC Değeri : ', roc_auc_score(Y_test, tahmin_test_ihtimal))

hassasiyet, duyarlılık, _ = precision_recall_curve(Y_test, tahmin_test_ihtimal)


print("Logartimik Kayıp (log-loss) : " , log_loss(Y_test, tahmin_test_ihtimal))
print("Hata Oranı                  : " , 1- accuracy_score(Y_test, tahmin_test))

C_değerleri = [0.1, 1, 10, 100]
dogruluk_df = pd.DataFrame(columns=['C_Değeri', 'Doğruluk'])

dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])

for c in C_değerleri:

    lr = LogisticRegression(penalty='l2', C=c, random_state=0)
    lr.fit(X_eğitim, Y_eğitim)
    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,
                                                    'Eğitim Doğruluğu': lr.score(X_eğitim, Y_eğitim),
                                                    'Test Doğruluğu': lr.score(X_test, Y_test)
                                                    }, ignore_index=True)
    X_eğitim, X_test, Y_eğitim, Y_test = train_test_split(X, Y, test_size=0.2, random_state=111)

    log_reg.fit(X_eğitim, Y_eğitim)
    fpr, tpr, thresholds = roc_curve(Y_test, tahmin_test_ihtimal)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, c={0} '.format(c))
    plt.show()

    hassasiyet, duyarlılık, a = precision_recall_curve(Y_test, tahmin_test_ihtimal)

    plt.plot(duyarlılık, hassasiyet)
    plt.title("Hassasiyet/Duyarlılık Eğrisi, c={0} ".format(c))
    plt.show()
print(dogruluk_değerleri)








