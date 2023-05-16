
### IMPORT LIBRARIES ###

import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import math
from adjustText import adjust_text
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from numpy import random
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import metrics
import statsmodels.api as sm

### FUNCTIONS ###

### FUNCTIONS ###
# Function: add labels by probabilities
def assign_labels_by_probabilities(df, scores_col, label_col, probs_col, quantile=0.85, num_samples=100):
    # Sort the dataframe by scores column in descending order
    annotated = df.sort_values(by=scores_col, ascending=False)
    annotated.reset_index(drop=True, inplace=True)

    # Assign probability of 0 to bottom whatever quantile of scores
    annotated.loc[annotated[scores_col] < annotated[scores_col].quantile(quantile), probs_col] = 0

    # Count the number of NaN values in the probabilities column - how many scores left
    num_nans = annotated[probs_col].isna().sum()

    # Write a linear function to assign increasing probabilities
    function = np.linspace(start=0.99, stop=0.01, num=num_nans)
    sum_func = np.sum(function)
    function = function/sum_func
    function = pd.Series(function)

    # Assign increasing probabilities to all NaNs
    annotated[probs_col].fillna(value=function, inplace=True)

    # Randomly select users based on assigned probabilities
    selected = random.choice(annotated["user_id"], size=num_samples,replace=False, p=annotated[probs_col])
    annotated[label_col] = 0
    annotated.loc[annotated['user_id'].isin(selected), label_col] = 1
    
    return annotated

# A function to remove protected characteristics and useless data
def drop_data_exp_2(df):
  labels_to_drop = ["user_id", "age", "gender", "education_level", "country", "test_run_id", "battery_id", "time_of_day", 
                    "model_A_scores", "model_B_scores", "Model_A_probabilities", "Model_B_probabilities"]
  clean = df.drop(labels_to_drop, axis = 1)
  return clean

# A function to train an SVM
def train_and_predict(name, X_train, X_test, y_train, y_test, kernel='poly'):
    # Define X and Y data
    name=name

    # Create a svm Classifier
    clf = svm.SVC(kernel=kernel, probability = True) # Polynomial Kernel

    # Train the model using the training sets
    model = clf.fit(X_train, y_train.values.ravel())

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)

    # Predict the probabilities for test dataset
    y_pred_proba = clf.predict_proba(X_test)

    # Change class probabilities into 2 flat numpy arrays
    array1 = y_pred_proba[:, 0].reshape(-1, 1).flatten()
    array2 = y_pred_proba[:, 1].reshape(-1, 1).flatten()

    # Append predictions to X_test dataframe
    X_eval = X_test.copy(deep=True)
    X_eval[f"Predicted_%s" % name] = y_pred

    # Append probability predictions to X_test dataframe
    X_eval[f"Prob_0_%s" % name] = array1
    X_eval[f"Prob_1_%s" % name] = array2

    # Mark which data was used for training
    X_tr = X_train.copy(deep = True)
    X_tr[f"Predicted_%s" % name] = "train"

    # Concatenate training and test data
    X_full = pd.concat([X_eval, X_tr])

    # Reset index and retain old index to be able to get back to sensitive data
    X_full = X_full.reset_index()

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Calculate precision
    precision = metrics.precision_score(y_test, y_pred)

    # Calculate recall
    recall = metrics.recall_score(y_test, y_pred)

    return accuracy, precision, recall, X_full, cm

def plot_data(data, protected_characteristic, colour_code):

  if protected_characteristic == 'age':
      bin_width= 1
      nbins = math.ceil((data["age"].max() - data["age"].min()) / bin_width)
      fig = px.histogram(data, x='age', nbins=nbins)
      fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))
      st.plotly_chart(fig, use_container_width=True)
      mean = data.loc[:, 'age'].mean().round(2)
      st.markdown(f'The mean age for this group is %s years.' % mean)

  elif protected_characteristic == 'education_level':
      data = data[protected_characteristic].value_counts().to_frame().reset_index()
      fig = px.bar(data, x=data.iloc[:,1], y=data.iloc[:,0], orientation='h',color=data.iloc[:,1])
      fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))
      fig.update_coloraxes(showscale=False)
      fig.update_layout(yaxis_title=None)
      fig.update_layout(xaxis_title=None)
      st.plotly_chart(fig, use_container_width=True)

  else:
      data = data[protected_characteristic].value_counts().to_frame().reset_index()
      fig = px.pie(data, values=data.iloc[:,1], names=data.iloc[:,0], color = data.iloc[:,0],
                    height=300, width=200, color_discrete_map=colour_code)
      fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))
      st.plotly_chart(fig, use_container_width=True)

def run_PCA(df, drop_1, retain_this, n):
  df_clean = df.drop(columns = [drop_1, retain_this, "index"])
  labels = list(df_clean.columns)
  pca = PCA(n_components=n)
  principalComponents = pca.fit_transform(df_clean)
  if n == 2:
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
  else:
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
  finalDf = pd.concat([principalDf, df[[retain_this]]], axis = 1)
  finalDf2 = finalDf.rename(columns = {retain_this : 'target'})
  coeff = np.transpose(pca.components_[0:2, :])
  return pca, finalDf2, labels, coeff, principalComponents

def create_confusion_matrix_heatmap(confusion_matrix):
  sns.set(font_scale=1.4)
  group_names = ['True Neg (TN)','False Pos (FP)','False Neg (FN)','True Pos (TP)']
  group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  fig, ax = plt.subplots()
  sns.heatmap(confusion_matrix, annot=labels, cmap='Blues', fmt='')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  st.write(fig)

def plot_conf_rates(confusion_matrix):
  TN = confusion_matrix[0,0]
  TP = confusion_matrix[1,1]
  FP = confusion_matrix[0,1]
  FN = confusion_matrix[1,0]


    # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # False discovery rate
  FDR = FP/(TP+FP)

  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)
  d = {'Measure': ['True Positive Rate', 'True Negative Rate', 'Positive Predictive Value', 'Negative Predictive Value', 'False Positive Rate', 'False Negative Rate', 'False Discovery Rate'], 
  'Equation' : ['TPR = TP/(TP+FN)', 'TNR = TN/(TN+FP)', 'PPV = TP/(TP+FP)', 'NPV = TN/(TN+FN)', 'FPR = FP/(FP+TN)', 'FNR = FN/(TP+FN)', 'FDR = FP/(TP+FP'], 
  'Score': [TPR, TNR, PPV, NPV, FPR, FNR, FDR]}
  st.table(d)
