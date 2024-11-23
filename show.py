import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,PowerTransformer
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub

raw = pd.read_excel('data.xlsx')
df = pd.read_csv('data_show.csv')



# Plotting the data by BMI
plt.figure(figsize=(10, 6))
sns.histplot(df['bmi'], bins=30, kde=True)
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.savefig('vis/bmi_distribution.png')

# Plotting a pie chart by class
# Mapping the values to labels
df['class'] = df['class'].map({0: 'non-overweight', 1: 'overweight', 2: 'obese'})

plt.figure(figsize=(8, 8))
df['class'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1')
plt.title('Weight Class Distribution')
plt.ylabel('')
plt.savefig('vis/class_distribution.png')
plt.show()

# # Plotting feature
# plt.figure(figsize=(10, 6))
# sns.histplot(df['act_sit_min'], bins=30, kde=True)
# plt.title('Distribution of feature')
# plt.xlabel('feature')
# plt.ylabel('Frequency')
# plt.show()