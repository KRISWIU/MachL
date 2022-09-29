import numpy as np
import pandas
from sklearn import datasets
import matplotlib . pyplot as plt
from sklearn.preprocessing import maxabs_scale

#1. Handwritten

digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]
N = len(digits_dataset_X)
digit_to_show = np.random.choice(range(N),1)[0]
print("Attributes:", digits_dataset_X[digit_to_show])
print("Class:", digits_dataset_y[digit_to_show])
plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8,8)))
plt.show()




#2. titanic

datasets = np.array(pandas.read_csv('titanic.csv'))
#we do not need name for this dataset cuz we have siblings aboard
x = datasets[:,[1,3,4,5,6,7]]
y = datasets[:,0]
for i in range(len(y)):
    #set male to 1 an female to 0
    if x[i][1] == 'female':
        x[i][1] = 0
    if x[i][1] == 'male':
        x[i][1] = 1
    #reset fare(i found out that male with fare<10 likely to have survive = 0 and fare is just decide where the passanger would sit in real life so it could be normalized as catagories): 0-10:0 ; 10-20:1; 20-30:2; 30-50:3; 50+:4
    if x[i][5]<10:
        x[i][5] = 0
    elif x[i][5]<20:
        x[i][5] = 1
    elif x[i][5]<30:
        x[i][5] = 2
    elif x[i][5]<50:
        x[i][5] = 3
    else:
        x[i][5] = 4
    #reset age to age/100
    x[i][2] = x[i][2]/100


#3. Loan

datasets = np.array(pandas.read_csv('loan.csv'))
x = datasets[:,[1,2,3,4,5,6,7,8,9,10,11,12]]#last column is for furter use of if the candidate have CoapplicantIncome
y = datasets[:,12]
maxApplicantIncome = 0
maxCoapplicantIncome = 0
maxLoanAmount = 0
for i in range(len(y)):
    if x[i][5]>maxApplicantIncome:
        maxApplicantIncome = x[i][5]
    if x[i][6]>maxCoapplicantIncome:
        maxCoapplicantIncome = x[i][6]
    if x[i][7]>maxLoanAmount:
        maxLoanAmount = x[i][7]

for i in range(len(y)):
    #reset value in y: Y:1, N:0
    if y[i] == 'Y':
        y[i] = 1
    else:
        y[i] = 0
    #for last column in x, urban:2, semiurban:1, rural:0
    if x[i][10] == 'Rural':
        x[i][10] = 0
    elif x[i][10] == 'Semiurban':
        x[i][10] = 1
    elif x[i][10] == 'Urban':
        x[i][10] = 2
    #set male to 1 an female to 0
    if x[i][0] == 'Female':
        x[i][0] = 0
    elif x[i][0] == 'Male':
        x[i][0] = 1
    #set second column in x, Yes = 1, No = 0
    if x[i][1] == 'No':
        x[i][1] = 0
    elif x[i][1] == 'Yes':
        x[i][1] = 1
    if x[i][2] == '1':
        x[i][2] = 1
    else:
        x[i][2] = 0
    #set 3rd column in x, Graduate = 1, UnderG = 0
    if x[i][3] == 'Graduate':
        x[i][3] = 1
    else:
        x[i][3] = 0
    #self employeed or not, E = 1, not E = 0
    if x[i][4] == 'No':
        x[i][4] = 0
    else:
        x[i][4] = 1
    #loan amount term usually 360 is the largerst amount so =loan/360
    x[i][8] = x[i][8]/360
    #Since there are many people with out CoapplicantIncome, so last column is set for if this person have CoapplicantIncome
    if x[i][7]>0:
        x[i][11] = 1
    else:
        x[i][11] = 0
    x[i][5] = x[i][5]/maxApplicantIncome
    x[i][6] = x[i][6]/maxCoapplicantIncome
    x[i][7] = x[i][7]/maxLoanAmount

#4. Parkinsons
datasets = np.array(pandas.read_csv('parkinsons.csv'))
x = datasets[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y = datasets[:,22]
#for first 3 column in x, x[i][0,1,2] = x[i][0,1,2]/max and HNR also to be processed like this and spread1 should be spread1/minspread1 since it is all negative
max1 = 0
max2 = 0
max3 = 0
maxhnr = 0
minspread1 = 100
for i in range(len(y)):
    if x[i][0]>max1:
        max1 = x[i][0]
    if x[i][1]>max2:
        max2 = x[i][1]
    if x[i][2]>max3:
        max3 = x[i][2]
    if x[i][15]>maxhnr:
        maxhnr = x[i][15]
    if x[i][18]<maxhnr:
        minspread1 = x[i][18]
for i in range(len(y)):
    x[i][0] = x[i][0]/max1
    x[i][1] = x[i][1]/max2
    x[i][2] = x[i][2]/max3
    x[i][15] = x[i][15]/maxhnr
    x[i][18] = x[i][18]/minspread1
print(x[0])