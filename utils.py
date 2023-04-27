
### IMPORT LIBRARIES ###

import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import math
from matplotlib_venn import venn3
from adjustText import adjust_text
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

### FUNCTIONS ###

def create_confusion_matrix_heatmap(confusion_matrix):
  sns.set(font_scale=1.4)
  group_names = ['True Neg (TN)','False Pos (FP)','False Neg (FN)','True Pos (TP)']
  group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.to_numpy().flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.to_numpy().flatten()/np.sum(confusion_matrix.to_numpy())]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  fig, ax = plt.subplots()
  sns.heatmap(confusion_matrix, annot=labels, cmap='Blues', fmt='')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  st.write(fig)

def plot_conf_rates(confusion_matrix):
  TN = confusion_matrix.iloc[0,0]
  TP = confusion_matrix.iloc[1,1]
  FP = confusion_matrix.iloc[0,1]
  FN = confusion_matrix.iloc[1,0]

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

def create_selectbox(title, options):
  selectbox = st.selectbox(title, options, index = 0)
  return selectbox

def run_PCA(df, drop_1, drop_2, retain_this, n):
  df_clean = df.drop(columns = [drop_1, drop_2, retain_this])
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

def plot_no_loadings(df):
  targets = [0, 1]
  markers = ["s","o"]
  colors = ['goldenrod', 'red']
  fig = plt.figure(figsize = (8,8))
  ax = fig.add_subplot(1,1,1) 
  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA', fontsize = 20)
  for target, color, marker in zip(targets, colors, markers):
    indicesToKeep = df['target'] == target
    plt.scatter(df.loc[indicesToKeep, 'principal component 1'], df.loc[indicesToKeep, 'principal component 2'], c = color, s = 20, alpha = 0.7, marker=marker)
  st.pyplot(fig)

def plot_loadings(pca, df, color_dict, coeff, labels):
  targets = [0, 1]
  markers = ["s","o"]
  colors = ['goldenrod', 'red']
  fig = plt.figure(figsize = (8,8))
  ax = fig.add_subplot(1,1,1) 
  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA', fontsize = 20)
  for target, color, marker in zip(targets,colors, markers):
    indicesToKeep = df['target'] == target
    plt.scatter(df.loc[indicesToKeep, 'principal component 1'], df.loc[indicesToKeep, 'principal component 2'], c = color, s = 20, alpha = 0.2, edgecolors = 'none', marker=marker)
  for i in range(coeff.shape[0]):
    plt.arrow(0, 0, coeff[i,0], coeff[i,1], width = 0.01, color=color_dict[i])
    texts = [plt.text(coeff[i, 0], coeff[i, 1], labels[i], ha='center', va='center', fontsize = 'medium', fontvariant = 'small-caps', fontweight = 'extra bold', color=color_dict[i]) for i in range(coeff.shape[0])]
    adjust_text(texts)

  plt.xlabel("PC{}".format(1))
  plt.ylabel("PC{}".format(2))
  plt.grid()
  st.pyplot(fig)