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
  st.markdown('''This section displays the distribution of scores assigned to each hypothetical employee, according to the values you set on the sliders, compared with the distribution of protected characteristics. These scores are then used to assign labels to the hypothetical employees, creating two distinct datasets to train the models. In essence, these scores explicitly and numerically mimic the viewpoints of hypothetical hiring managers A and B when deciding who to label as "top employees."''')
  st.markdown('''Just as in the original NCPT dataset analysis, the scores obtained by participants in the cognitive games decline with age. This observation provides insight into the potential issue of ageism inherent in the use of gamified hiring processes.''')
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
  st.markdown('''On this page, you can see the distribution of the dataset labels which were assigned based on the scores calculated from the slider values you selected previously. Principal Components Analysis, or PCA, is a technique often used to analyse and subsequently visualise datasets where there are many features per single example. This is the case with the NCPT dataset used in our simulator. Specifically, the battery which we used has 11 features per single example, the example being the player of cognitive games, and, in our metaphor, a hypothetical employee or job candidate. It is impossible to plot 11 dimensions, and PCA allows for the visualisation of multidimensional data, while also preserving as much information as possible.''')
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
      
    st.markdown('''These plots show the reduction of 11 dimensions (11 subtest results) to 3 dimensions. At the top, you can see how much information, or "Total Variance", has been preserved. Note that for both datasets, A and B, different points are labelled "1" or "0". This shows that the two datasets represent the two different target variable definitions which were created by you previously. The plots are interactive - zoom in to explore in detail.''')

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
    st.markdown('''On this plot, PCA component loadings can be explored. These facilitate the understanding of how much each variable (which there are 11 of) contributes to a particular principal component. Here, the 11 variables were reduced to 2 components, which are labelled PC1 and PC2 on the x and y axes. The magnitude of the loading (here displayed as an arrow per each variable) indicates how strong the relationship between the variable and the component is. Therefore, the longer the arrow, the stronger the relationship between that component and that variable. The loading's sign can be positive or negative. This indicates whether the principal component and that variable are positively or negatively correlated. We can see that multiple variables are positively correlated with PC2. Two variables, episodic verbal learning and delayed recall are negatively correlated with both of the components, which means that the variance in them is not well represented by these two components.''')


def model_out(full_df):
  st.markdown('''This section highlights the discrepancies between your two models when presented with the same pool of new, previously unseen candidates to label. Specifically, you'll be investigating the candidates assigned a "1" label by both models. These individuals would be those considered for a job interview or chosen for the role, according to your defined target variable.''')
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
      
  
  st.markdown('''In this section, you're comparing the model's selections concerning four protected characteristics: age, gender, education level, and country. You can visualize these differences in two ways: "Absolute" or "Proportional".''')
  st.markdown('''"Absolute" representation gives you the raw numbers or percentages of each characteristic chosen. For instance, if the model labeled 5 female candidates and 5 male candidates as "1", the "Absolute" outcome will display as 50% for both genders."Proportional" representation, on the other hand, shows the percentage of a group selected by the model relative to the total number of that group in the input data. For example, if the model evaluated 100 male candidates and selected 5, you will see a 5% representation. If it evaluated 200 female candidates and selected 5, it will show a 2.5% representation.''')
  st.markdown('''If you encounter empty categories in the "Proportional" view, this indicates that while candidates from these categories were evaluated, none were labeled as "1". Hence, their proportional representation amounts to 0%.''')

def dataframe_out(full_df):
  selectbox_M = st.selectbox('Choose which model output to rank by', pred_dict.keys())

  # Select data
  data = full_df.loc[full_df[pred_dict[selectbox_M]] == 1]
  data = data.sort_values(by = prob_dict[selectbox_M], ascending = False)
  data = data[['Candidate ID','Prob_1_A', 'Prob_1_B', 'Predicted_A', 'Predicted_B']]
  data = data.rename(columns={"Prob_1_A": "Ranking, model A", "Prob_1_B": "Ranking, model B", "Predicted_A": "Predicted label A", "Predicted_B": "Predicted label B"})
  data.index = np.arange(1, len(data) + 1)

  st.table(data.style.background_gradient(subset = ["Ranking, model A", "Ranking, model B"], axis=0, vmin=0.40).highlight_max(color = '#FFCD9B', subset = ["Predicted label A", "Predicted label B"], axis=0))
  
  st.markdown("""In this section, you can review the data for all candidates labeled "1" by the selected model, found at the top of the page. Simultaneously, you can observe the labels assigned to these same candidates by the other model. It's likely that there will be instances where candidates chosen by one model weren't selected by the other. Candidates labeled "1" are highlighted in orange in the "Predicted label A" and "Predicted label B" columns.""")
  st.markdown('''In addition to this, you can see the probability with which each candidate was labeled "1". The intensity of the blue color indicates the candidate's ranking position - a darker blue represents a higher ranking (with 1 being the maximum and 0 the minimum). You may notice that some candidates highly ranked by one model may be ranked significantly lower by the other model.''')
  
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
    st.markdown('''This Venn Diagram visualizes the number of candidates chosen by both models. It's likely that some candidates will be selected by both models, while others may be chosen by only one model. If we consider Model A as the decision of one hiring manager and Model B as another's, it's easy to see how the selection outcome varies depending on the decision-maker. Some candidates may get the opportunity to be hired, while others might not. This serves as an illustration of the inherent arbitrariness in defining the target variable when dealing with highly subjective outcomes.''')
    st.markdown('''For instance, it's straightforward to define a target variable in a classification problem like distinguishing dragonflies from butterflies, where there's little room for ambiguity. However, defining what makes a 'good' employee is far more challenging due to its subjective nature.''')
    


def model_vis(full_df):
  st.markdown('''In this section, you can visualise the demographics of the different subgroups of the data. Firstly, you can see the demographic characteristics of the candidates who have positive labels ("1") and negative labels ("0") which were assigned based on the scores calculated from the slider values you selected previously. Then, you can visualise the demographic distributions of the data which was used for training and evaluation of the models.''')
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
    st.markdown('''You are visualising the demographic composition of those hypothetical employees who were assigned labels "1" or "0" based on your definitions of the target variables. You might see differences in proportions of genders between the two models for the positive labels, as well as a major difference in the age between the positive and negative labels. Visualising the labels in this manner before training the model can help understand and mitigate differences in demographic representation in the modelling outcomes. Likely, if all candidates labelled "1" were in younger age groups, the candidates selected by the model at the deployment stage will also be in younger age groups. Moreover, target variable definition affects the proportional representation. Having defined two target variables, one can choose the dataset and the model which offers more proportional representation.''')


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
  
    st.markdown('''To train a machine learning model, the data has to be split into two different sets. The first set is the training data, which will be used to teach the model the relationships between the input features (11 subtest results) and the corresponding labels ("0" and "1", assigned based on your definitions of target variables and the values you chose for the sliders). The second set is the test data, or evaluation data. It is used to assess the performance of the model. This is the data which is used to plot the confusion matrices and calculate the model metrics which you saw at the bottom of the "Define the target variable" page. This is also the data whose features you can explore in "Modelling outcomes". It is important that the training and testing data are balanced. Here, you can compare the demographic composition of the training and evaluation data. The training and evaluation datasets compositions were the same and contained the same candidates and same features for both models A and B. However, the labels for each dataset were different and based on what you selected in "Define target variable".''')

def filter_for_protected(data):
  st.markdown('''This section shows equivalent metrics and confusion matrices to those presented at the bottom of "Define target variable", except applied to specific subgroups of the evaluation data, or the "hypothetical candidates" as assessed by the different models. Sometimes, the overall model metrics can be deceptive when it comes to predicting the results for different groups in consideration. Ideally, for our models, the varying model metrics would be similar across different groups, which would indicate that the overall model performance is reflected in how this model performs for a given group. It is often not the case, and it is likely that you will see that models A and B perform differently when it comes to those metrics. Even the same model can have different metrics for different subgroups.''')
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