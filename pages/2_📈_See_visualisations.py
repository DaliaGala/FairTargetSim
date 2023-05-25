#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:30:56 2023

@author: daliagala
"""

### IMPORT LIBRARIES ###
import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn2
from sklearn.metrics import confusion_matrix
from utils import display_proportional, plot_data, run_PCA, create_confusion_matrix_heatmap, plot_conf_rates


### DICTIONARIES AND CONSTANTS###

colours_education = {
          'Some high school' : 'indigo',
          'High school diploma / GED' : '#7ae99e',
          'Some college' : '#0a68c9',
          'College degree': '#80c4fa',
          'Professional degree': '#f48508',
          "Master's degree" : '#2fa493',
          'Ph.D.' : '#f2a3a1',
          "Associate's degree" : '#fbcc66',
          'Other' : '#fa322f'
        }

colours_country = {
          'AU' : '#7ae99e',
          'US': '#80c4fa',
          'NZ': '#2fa493',
          'CA' : '#fbcc66'
        }

colours_gender = {
          'f' : '#7ae99e',
          'm': '#2fa493'
        }

characteristic_dict = {
          'gender' : colours_gender,
          'education level' : colours_education,
          'country' : colours_country,
          'age' : 'indigo'
        }

pred_dict = {
          'Model A' : 'Predicted_A',
          'Model B' : 'Predicted_B'
        }

prob_dict = {
          'Model A' : 'Prob_1_A',
          'Model B' : 'Prob_1_B'
        }

model_dict = {
          'Model A' : 'Model_A_label',
          'Model B' : 'Model_B_label'
        }

### DEFINE ALL SUB-PAGES AS FUNCTIONS TO CALL ###

def mod_prop(cmA, cmB):
  st.markdown('''On this page you can see output metrics for each model. They include model [confusion matrices](https://en.wikipedia.org/wiki/Confusion_matrix) 
  and tables containing model [metrics](https://en.wikipedia.org/wiki/Precision_and_recall).''')

  row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))

  with row1_1:
    st.subheader("Model A confusion matrix")
    create_confusion_matrix_heatmap(cmA)
    plot_conf_rates(cmA)

  with row1_2:
    st.subheader("Model B confusion matrix")
    create_confusion_matrix_heatmap(cmB)
    plot_conf_rates(cmB)

def model_scores(dataframe):
  st.markdown('''This section visualises the score distribution assigned to each hypothetical employee based on the values you selected for each slider versus the protected characteristic distribution. These scores are then used to 
  assign labels to the hypothetical employees. Therefore, two different datasets are created based on which we can train models. These scores emulate, in an **explicit and numerical** manner,
  the opinions of the hypothetical hiring managers A and B when they try to decide who to select as "top employees". Much like in the case of the original [**NCPT dataset**](https://www.nature.com/articles/s41597-022-01872-8) analysis,
  the scores obtained by the players of the cognitive games decline with age. This presents an insight about the usage of gamified hiring where its very nature could be considered ageist.''')
  # Create a selectbox to choose a protected characteristic to explore
  plot_radio = st.radio('Characteristic to explore', characteristic_dict.keys(), horizontal=True)
  row2_space1, row2_1, row2_space2 = st.columns((0.1, 5, 0.1))

  with row2_1:
    data = dataframe[["model_A_scores", "model_B_scores", plot_radio]]

    if plot_radio == "age":
      selectbox_Mod = st.selectbox('Choose model', ("Model A", "Model B"))
      if selectbox_Mod == "Model A":
        fig = px.scatter(data, x=data['age'], y="model_A_scores", trendline="ols")
        fig.update_layout(yaxis_title="Dataset A scores")
        st.write(fig)
      else:
        fig = px.scatter(data, x=data['age'], y="model_B_scores", trendline="ols")
        fig.update_layout(yaxis_title="Dataset B scores")
        st.write(fig)
            
    else:
      
      fig = go.Figure(layout=go.Layout(height=700, width=900))

      fig.add_trace(go.Box(
          y = data["model_A_scores"],
          x = data[plot_radio],
          name = 'Dataset A Scores',
          marker_color = '#3D9970'
      ))
      fig.add_trace(go.Box(
          y = data["model_B_scores"],
          x = data[plot_radio],
          name = 'Dataset B Scores',
          marker_color='#FF4136'
      ))

      fig.update_layout(
          yaxis_title='dataset scores',
          boxmode='group' # group together boxes of the different traces for each value of x
      )
      st.write(fig)

def PCA_general(full_df, dataframe_PCA):
  st.subheader("Principal components analysis")
  choice = st.radio("What would you like to explore?", ("PCAs", "Components loading"))
  pcaA, dfA, labelsA, coeffA, componentsA = run_PCA(dataframe_PCA, 'Model_B_label', 'Model_A_label', 2)
  pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 2)
  loadings = pcaB.components_.T * np.sqrt(pcaB.explained_variance_)
  total_var = pcaA.explained_variance_ratio_.sum() * 100
  dfA = dfA.rename(columns={'target': 'Dataset A'}).reset_index()
  dfB = dfB.rename(columns={'target': 'Dataset B'}).reset_index()
  df_all = pd.merge(dfA, dfB[['index', 'Dataset B']], on='index', how='left')

  conditions = [
      (df_all['Dataset A'] == 1) & (df_all['Dataset B'] == 0),
      (df_all['Dataset B'] == 1) & (df_all['Dataset A'] == 0),
      (df_all['Dataset A'] == 1) & (df_all['Dataset B'] == 1),
      (df_all['Dataset A'] == 0) & (df_all['Dataset B'] == 0)]
      
  values = ['Selected A', 'Selected B', 'Selected both', 'Not selected']
  df_all['All'] = np.select(conditions, values)

  df_all = df_all.drop(["index"], axis = 1)
  df_all.All=pd.Categorical(df_all.All,categories=['Not selected', 'Selected A', 'Selected B', 'Selected both'])
  df_all=df_all.sort_values('All')

  selections_dict = {0: 'Not selected', 1: 'Selected'}
  df_all = df_all.replace({"Dataset A": selections_dict, "Dataset B": selections_dict})

  color_dict_sel = {'Not selected': '#3366CC', 'Selected': 'whitesmoke'}

  if "pca_df" not in st.session_state:
    st.session_state.pca_df = df_all

  if choice == "PCAs":
    c1, c2 = st.columns(2)
    with c1:
      fig = px.scatter(st.session_state.pca_df, 
                      x=st.session_state.pca_df['principal component 1'].astype(str),
                      y=st.session_state.pca_df['principal component 2'].astype(str),
                      title='Dataset A PCA',
                      labels={'0': 'PC 1', '1': 'PC 2'},
                      color=st.session_state.pca_df['Dataset A'],
                      color_discrete_map=color_dict_sel)
      fig.update_traces(marker_size = 8)
      st.plotly_chart(fig, use_container_width=True)
    with c2:
      fig = px.scatter(st.session_state.pca_df, 
                      x=st.session_state.pca_df['principal component 1'].astype(str),
                      y=st.session_state.pca_df['principal component 2'].astype(str),
                      title='Dataset B PCA',
                      labels={'0': 'PC 1', '1': 'PC 2'},
                      color=st.session_state.pca_df['Dataset B'],
                      color_discrete_map=color_dict_sel)
      fig.update_traces(marker_size = 8)
      st.plotly_chart(fig, use_container_width=True)
      
    st.markdown('''Now you can see the distribution of the dataset labels which were assigned based on the scores calculated from the slider values above. 
      [Principal Components Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), or PCA, is a technique often used to analyse and subsequently visualise datasets 
      where there are many features per single example. This is the case with the NCPT dataset used in our simulator. Specifically, the battery which we used has 11 features per single 
      example, the example being the player of cognitive games, and, in our metaphor, a hypothetical employee or job candidate. It is impossible to plot 11 dimensions, and PCA allows 
      for the visualisation of multidimensional data, while also preserving as much information as possible. The plots below shows the reduction of 11 dimensions (11 subtest results) to 
      3 dimensions. At the top, you can see how much information, or "Total Variance", has been preserved. Note that for both datasets, A and B, different points are labelled "1" or "0". 
      This shows that the two datasets represent the two different target variable definitions which were created by you above. The plots are interactive - zoom in to explore in detail.''')

    pcaA, dfA, labelsA, coeffA, componentsA = run_PCA(dataframe_PCA, 'Model_B_label', 'Model_A_label', 2)
    pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 2)
    loadings = pcaB.components_.T * np.sqrt(pcaB.explained_variance_)
    total_var = pcaA.explained_variance_ratio_.sum() * 100
    dfA = dfA.rename(columns={'target': 'Dataset A'}).reset_index()
    dfB = dfB.rename(columns={'target': 'Dataset B'}).reset_index()
    df_all = pd.merge(dfA, dfB[['index', 'Dataset B']], on='index', how='left')

    conditions = [
        (df_all['Dataset A'] == 1) & (df_all['Dataset B'] == 0),
        (df_all['Dataset B'] == 1) & (df_all['Dataset A'] == 0),
        (df_all['Dataset A'] == 1) & (df_all['Dataset B'] == 1),
        (df_all['Dataset A'] == 0) & (df_all['Dataset B'] == 0)]
        
    values = ['Selected A', 'Selected B', 'Selected both', 'Not selected']
    df_all['All'] = np.select(conditions, values)

    df_all = df_all.drop(["index"], axis = 1)
    df_all.All=pd.Categorical(df_all.All,categories=['Not selected', 'Selected A', 'Selected B', 'Selected both'])
    df_all=df_all.sort_values('All')

    selections_dict = {0: 'Not selected', 1: 'Selected'}
    df_all = df_all.replace({"Dataset A": selections_dict, "Dataset B": selections_dict})

    if "pca_df" not in st.session_state:
      st.session_state.pca_df = df_all

    fig = px.scatter(st.session_state.pca_df, 
                      x=st.session_state.pca_df['principal component 1'],
                      y=st.session_state.pca_df['principal component 2'],
                      title=f'Total Explained Variance: {total_var:.2f}%',
                      color=st.session_state.pca_df["All"], 
                      width = 800, height = 800,
                      color_discrete_sequence=px.colors.qualitative.Safe,
                      opacity = 0.95)
      
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
      )
    fig.update_traces(marker_size = 10)
    st.plotly_chart(fig)

  if choice == "Components loading":
    
    fig = px.scatter(st.session_state.pca_df, 
                    x=st.session_state.pca_df['principal component 1'],
                    y=st.session_state.pca_df['principal component 2'],
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    color=st.session_state.pca_df["All"],
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    opacity = 0.95)
    
    for i, feature in enumerate(labelsA):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0],
            y=loadings[i, 1],
            showarrow=True,
            arrowsize=2,
            arrowhead=2,
            xanchor="right",
            yanchor="top"
        )

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5
        )
    
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
      )
    fig.update_traces(marker_size = 10)
    st.plotly_chart(fig, use_container_width = True)


def model_out(full_df):
  st.markdown('''This section shows you the differences between your two models when they are given the same set of previously unseen candidates to assign labels to. You are exploring those candidates who were given label "1" by the models,
  therefore, they would be the ones offered the job interview or selected by the position based on your target variable definition.''')
  # Create a selectbox to choose a protected characteristic to explore
  selectbox = st.selectbox('Characteristic to explore', characteristic_dict.keys())
  representation = st.selectbox("Representation", ("absolute", "proportional"))
  row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))
  with row1_1:
    st.subheader("Candidates selected by model A")

    if representation == "absolute":
      # Select predicted data ==1
      data = full_df.loc[full_df['Predicted_A'] == 1]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox, characteristic_dict[selectbox])
    else:
      display_proportional(full_df, selectbox, 'Predicted_A')
      
  with row1_2:
    st.subheader("Candidates selected by model B")
    
    if representation == "absolute":
      # Select predicted data ==1
      data = full_df.loc[full_df['Predicted_B'] == 1]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox, characteristic_dict[selectbox])

    else:
      display_proportional(full_df, selectbox,'Predicted_B')
      
  
  st.markdown("""Here, you compare how the models choices vary with respect to representation of the four protected characteristics, 
  which are: age, gender, education level and country. You can visualise the difference either as "Absolute" or "Proportional". "Absolute" 
  shows the absolute numbers or percentage selected per each characteristic. Therefore, if the model assigned label "1" to 5 female 
  candidates and 5 male candidates, you will see that the "Absolute" outcome is 50% and 50%. "Proportional" shows what percentage of this 
  group was selected by the model as a proportion of the input proportion of this group. For example: if 100 male candidates were evaluated 
  by the model and 5 were selected, you will see 5% representation, and if 200 female candidates were evaluated, and 5 were selected, you will 
  see 2.5% representation. If you see empty categories in the "Proportional" view, this means that representatives of these categories were 
  evaluated by the model, but none of them were labelled "1", therefore, their proportional representation is 0%.""")

def dataframe_out(full_df):
  selectbox_M = st.selectbox('Choose which model output to rank by', pred_dict.keys())

  # Select data
  data = full_df.loc[full_df[pred_dict[selectbox_M]] == 1]
  data = data.sort_values(by = prob_dict[selectbox_M], ascending = False)
  data = data[['Candidate ID','Prob_1_A', 'Prob_1_B', 'Predicted_A', 'Predicted_B']]
  data = data.rename(columns={"Prob_1_A": "Ranking, model A", "Prob_1_B": "Ranking, model B", "Predicted_A": "Predicted label A", "Predicted_B": "Predicted label B"})
  data.index = np.arange(1, len(data) + 1)

  st.table(data.style.background_gradient(subset = ["Ranking, model A", "Ranking, model B"], axis=0, vmin=0.40).highlight_max(color = '#FFCD9B', subset = ["Predicted label A", "Predicted label B"], axis=0))
  
  st.markdown("""Here, you can see the data for all candidates labelled "1" by the model which you choose at the top. You will simultaneously 
  see what the same candidates were labelled as by the other model. It is likely that you will see that some candidates chosen by one model were 
  not chosen by the second model. Candidates labelled "1" are highlighted in orange in the columns "Predicted label A" and "Predicted label B". You will also see the probability 
  with which the candidates were labelled "1". The darker the colour blue, the higher to the top of the ranking the given candidate was (the maximum being 1, and the minimum being 0).
  You might see that some candidates ranked very highly for one of the models were ranked 
  much lower for the other model.""")

def venn_diagram(full_df):
  row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
  with row2_1:
    fig, ax = plt.subplots()

    list_A = full_df.loc[full_df['Predicted_A'] == 1, 'Candidate ID'].astype(int)
    list_B = full_df.loc[full_df['Predicted_B'] == 1, 'Candidate ID'].astype(int)
    set1 = set(list_A)
    set2 = set(list_B)

    venn2([set1, set2], ('Model A', 'Model B'), ax=ax)
    st.pyplot(fig)

  with row2_2:
    st.markdown('''This Venn Diagram presents the number of candidates which were selected by both models. It is likely that some candidates were selected 
    by both models, while others were selected specifically by only one model. If we imagine that model A represents the decision of one hiring manager, and model
    B represents the decision of another one, it is easy to understand that, depending on the circumstances, some candidates have the chance to be hired, and
    some don't. This exemplifies the arbitrary nature of target variable definition in the context of very subjective target variables. It is easy to 
    define a target variable when the classification problem concerns distinguishing dragonflies from butterflies. One is a dragonfly, and another is a butterfly,
    and there is not much space for uncertainty in this classification. It is much harder to define what a good employee is because this classification is subjective.''')


def model_vis(full_df):
  st.markdown('''In this section, you can visualise the demographics of the different subgroups of the data. Firstly, you can see the demographic characteristics
  of the candidates who have positive labels ("1") and negative labels ("0") which were assigned based on the scores calculated from the slider values you selected 
  previously. Then, you can visualise the demographic distributions of the data which was used for training and evaluation of the models.''')
  choice = st.radio("Choose what to explore", ("Positive and negative labels", "Training and evaluation data"), horizontal=True)
  if choice == "Positive and negative labels":
    # Create a selectbox to choose a protected characteristic to explore
    selectbox_Lab = st.selectbox('Label to visualise', ('positive labels', 'negative labels'))

    # Create a selectbox to choose a protected characteristic to explore
    selectbox_Char = st.selectbox('Characteristic to explore', characteristic_dict.keys())

    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))

    with row2_1:
      st.subheader("Dataset A")

      # Select test data
      if selectbox_Lab == 'positive labels':
        data = full_df.loc[full_df['Model_A_label'] == 1]
      else:
        data = full_df.loc[full_df['Model_A_label'] == 0]
      
      # Use function plot_data to plot selected data
      plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])


    with row2_2:
      st.subheader("Dataset B")

      # Select test data
      if selectbox_Lab == 'positive labels':
        data = full_df.loc[full_df['Model_B_label'] == 1]
      else:
        data = full_df.loc[full_df['Model_B_label'] == 0]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])
    st.markdown('''You are visualising the demographic composition of those hypothetical employees who were assigned labels "1" or "0" based on your definitions of the
    target variables. You might see differences in proportions of genders between the two models for the positive labels, as well as a major difference in the age between the positive and negative labels. 
    Visualising the labels in this manner before training the model can help understand and mitigate differences in demographic representation in the modelling outcomes. Likely,
    if all candidates labelled "1" were in younger age groups, the candidates selected by the model at the deployment stage will also be in younger age groups. Moreover, target
    variable definition affects the proportional representation. Having defined two target variables, one can choose the dataset and the model which offers more proportional representation.''')


  if choice == "Training and evaluation data":
    # Create a selectbox to choose a protected characteristic to explore
    selectbox = st.selectbox('Charactertistic to explore', characteristic_dict.keys())
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    # Plot training data
    with row1_1:
      st.subheader("Training data")

      # Select train data
      train = full_df.loc[full_df["Predicted_A"] == "train"]

      # Use function plot_data to plot selected data
      plot_data(train, selectbox, characteristic_dict[selectbox])

    # Plot test data

    with row1_2:
      st.subheader("Test data")

      # Select test data
      test = full_df.loc[full_df["Predicted_A"] != "train"]

      # Use function plot_data to plot selected data
      plot_data(test, selectbox, characteristic_dict[selectbox])
  
    st.markdown('''To train a machine learning model, the data has to be split into two different sets. The first set is the training data, which will be used to teach the model the relationships between the input features (11 subtest results)
    and the corresponding labels ("0" and "1", assigned based on your definitions of target variables and the values you chose for the sliders). The second set is the test data, or evaluation data. It is used to assess the performance of the model. 
    This is the data which is used to plot the confusion matrices and calculate the model metrics which you saw at the bottom of the "Define the target variable" page. This is also the data whose features you can explore in "Modelling outcomes".
    It is important that the training and testing data are balanced. Here, you can compare the demographic composition of the training and evaluation data. The training and evaluation datasets compositions were the same and contained the same 
    candidates and same features for both models A and B. However, the labels for each dataset were different and based on what you selected in "Define target variable".''')

def filter_for_protected(data):
  st.markdown('''This section shows equivalent metrics and confsion matrices to those presented at the bottom of "Define target variable", except applied to specific subgroups of the evaluation data, or the "hypothetical candidates"
  as assessed by the different models. Sometimes, the overall model metrics can be deceptive when it comes to predicting the results for different groups in consideration. Ideally, for our models, the varying model metrics 
  would be similar across different groups, which would indicate that the overall model performance is reflected in how this model performs for a given group. It is often not the case, and it is likely that you will see that models A
  and B perform differently when it comes to those metrics. Even the same model can have different metrics for different subgroups.''')
  model = st.selectbox('Choose which model outputs to assess', pred_dict.keys())
  test = data.loc[data[pred_dict[model]] != "train"]

  selectbox_Char = st.selectbox('Characteristic to explore', characteristic_dict.keys())
  if selectbox_Char == 'age':
    bins= [18,20,30,40,50,60,70,80,90]
    labels = ['18-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90']
    test['age_bins'] = pd.cut(test['age'], bins=bins, labels=labels, right=False)
    selectbox_Char = 'age_bins'
  which_group = st.selectbox('Which group?', test[selectbox_Char].unique())
  rslt_df = test[test[selectbox_Char] == which_group] 
  y_true = [int(numeric_string) for numeric_string in rslt_df[model_dict[model]]]
  y_pred = [int(numeric_string) for numeric_string in rslt_df[pred_dict[model]]]
  cm = confusion_matrix(y_true, y_pred)
  if cm.shape == (1,1):
    cm = np.array([[cm[0, 0], 0], [0, 0]])
  row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))
  with row1_1:
    create_confusion_matrix_heatmap(cm)
  with row1_2:
    plot_conf_rates(cm)
    
    

def data_plot(key1, key2, key3, key4):
  st.title('''Compare the models you trained''')
  if key1 not in st.session_state:
    st.error('Cannot train the models if you do not define the target variables. Go to "Define target variable"!', icon="🚨")
  else:
    tab1, tab2 = st.tabs(["Demographics", "Metrics"])
    with tab1:
      dataframe = st.session_state[key1]
      clean_data = st.session_state[key2]
      page_radio = st.radio('What demographics would you like to explore?', ("Modelling outcomes", "Input scores", "Label distribution", "Outcomes per group"), horizontal=True)
      if page_radio == "Input scores":
        model_scores(dataframe)
      if page_radio == "Modelling outcomes":
        model_out(dataframe)
      if page_radio == "Label distribution":
        model_vis(dataframe)
      if page_radio == "Outcomes per group":
        filter_for_protected(dataframe)
    with tab2:
      dataframe = st.session_state[key1]
      clean_data = st.session_state[key2]
      cmA = st.session_state[key3]
      cmB = st.session_state[key4]
      metrics_radio = st.radio('What metrics would you like to see?', ("Labelled dataframe", "Venn Diagram", "Model confusion matrices", "Principal Component Analysis"), horizontal=True)
      if metrics_radio == "Labelled dataframe":
        dataframe_out(dataframe)
      if metrics_radio == "Venn Diagram":
        venn_diagram(dataframe)
      if metrics_radio == "Model confusion matrices":
        mod_prop(cmA, cmB)
      if metrics_radio == "Principal Component Analysis":
        PCA_general(dataframe, clean_data)
    
data_plot('complete_df', 'clean_df', 'cm_A', 'cm_B')