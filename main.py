import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse

import kagglehub

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vis import *

df = pd.read_csv('data_processed.csv')



















