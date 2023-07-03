#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:16:50 2023

@author: daliagala
"""

### LIBRARIES ###
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from utils import assign_labels_by_probabilities, drop_data, train_and_predict

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')

hide_st_style = """
            <style>
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

### IMPORT DATA FILES ###
dataframe = pd.read_csv('./data/dataframe.csv')
dataframe = dataframe.drop(["Unnamed: 0"], axis = 1)
dataframe = dataframe.rename(columns={"education_level": "education level"})

### DICTIONARIES AND CONSTANTS###
groups = ["attention", "reasoning", "memory", "behavioural restraint", "information processing speed"]

groups_dict = {
          'Divided Visual Attention' : 'attention',
          'Forward Memory Span': 'memory',
          'Arithmetic Reasoning' : 'reasoning',
          'Grammatical Reasoning' : 'reasoning',
          'Go/No go' : 'behavioural restraint',
          'Reverse_Memory_Span' : 'memory',
          'Verbal List Learning' : 'memory',
          'Delayed Verbal List Learning' : 'memory',
          'Digit Symbol Coding' : 'information processing speed',
          'Trail Making Part A' : 'information processing speed',
          'Trail Making Part B' : 'information processing speed'
        }

education_dict = {
    1: 'Some high school',
    2: 'High school diploma / GED',
    3: 'Some college',
    4: 'College degree',
    5: 'Professional degree',
    6: "Master's degree",
    7: 'Ph.D.',
    8: "Associate's degree",
    99: 'Other'
  }

df_keys_dict = {
          'Divided Visual Attention' :'divided_visual_attention',
          'Forward Memory Span' :'forward_memory_span',
          'Arithmetic Reasoning' : 'arithmetic_problem_solving',
          'Grammatical Reasoning'  : 'logical_reasoning',
          'Go/No go': 'adaptive_behaviour_response_inhibition',
          'Reverse_Memory_Span' : 'reverse_memory_span',
          'Verbal List Learning': 'episodic_verbal_learning',
          'Delayed Verbal List Learning': 'delayed_recall',
          'Digit Symbol Coding': 'abstract_symbol_processing_speed',
          'Trail Making Part A' :'numerical_info_processing_speed',
          'Trail Making Part B': 'numerical_and_lexical_info_processing_speed'
        }


### CREATE THE "TARGET VARIABLE DEFINITION" PAGE ###
st.title('Target variable definition')

st.markdown('''On this page, we invite you to imagine that you are hiring for a certain role. Using the sliders below, you will specify two different notions of a “good employee” for that role—two different target variables. Once you’re done, the simulator will build two models, one for each of your target variable definitions. You can visualize these datasets and models—and their effects on fairness and overall features of the models and data—in the Visualize the Results page.''')

st.markdown('''You specify the notions of different employees by assigning weights of importance to cognitive characteristics that a “good” employee would have for the role. The cognitive test data that you’ll be working with comes from real-world people, mirroring an increasing number of hiring algorithms that are based cognitive tests ([(Wilson, et al. 2021)](https://dl.acm.org/doi/10.1145/3442188.3445928)).''')

st.markdown('''We have pre-set the weights below to reflect two different conceptions of a “good” employee: one conception that emphasizes attentiveness and numerical skills (A) and another conception that emphasizes interpersonal skills and memory (B). If you want to, you can change the slider values as you see fit.''')

st.markdown('''After you’ve set the slider values, click :red[“Assign labels and train your models”]. The simulator will then label certain individuals as "good employees"—in other words it will assign class labels "1" for successful and "0" for not good, based on your selections. Then, the simulator will build the two models.''')

st.markdown('''To learn more about target variable definition and hiring models based in cognitive tests, :green["See explanation"] below.''')

with st.expander("See explanation"):
    
    st.markdown('''The models that the simulator builds are of a kind increasingly used in hiring software: companies will have applicants play games that test for different kinds of cognitive ability (like reasoning or memory). Then hiring software will be built to predict which applicants will be successful based on which cognitive characteristics they have. What cognitive characteristics make for a successful employee? This will depend on what role is being hired. And it will also depend on how one defines “successful employee.”''')
    
    st.markdown('''In the real world, “successful employee” is defined for these kinds of hiring models in the following way. Managers select a group of current employees that they consider to be successful; this group of employees plays the cognitive test games. A model is then trained to identify applicants who share cognitive characteristics with the current employees that are considered successful. The target variable of “successful employee” is thus defined in terms of comparison to certain people who are deemed successful. ''')

    st.markdown('''One will get different target variables if one deems different current employees as the successful ones. And, as we discussed in the Home page (and as we explain more in the Putting the Idea into Practice page), there will likely be disagreement between managers about which employees are successful. For instance, a manager who values attentiveness and numerical skills will deem different employees “successful” than a manager who values interpersonal skills and memory. Even when different managers roughly share their sensibilities in what characteristics make for a successful employee, there may still be different, equally good ways to “weight” the importance of the various characteristics.''')
    
    st.markdown('''In the real world, the cognitive characteristics shared by those considered successful employees is implicit. Companies do not first identify the cognitive characteristics that make for a successful employee; rather, they identify employees who they consider successful, and then the hiring model works backwards to identify what characteristics these employees share.''')
    
    st.markdown('''In our simulator, the cognitive characteristics shared by “good” employees are explicit. You assign different weights—using the sliders—to the cognitive characteristics you think are more or less important in a good employee (for the role you’re considering). To illustrate how different target variables have different effects on fairness and overall model attributes, you’ll define “good” employee in two ways. (We’ve made the cognitive characteristics explicit both so you can see the point of different target variable definitions more clearly, and because of limitations of the data that we’re working with.)''')

    st.markdown('''The cognitive characteristics that the simulator works with are from one of datasets of the [NeuroCognitive Performance Test](https://www.nature.com/articles/s41597-022-01872-8). This dataset has eleven different tests which we have grouped into five categories: ''')
    
    st.markdown('''The cognitive characteristics that you can give different weights are taken from one of the datasets from the NeuroCognitive Performance Test ([**NCPT dataset**](https://www.nature.com/articles/s41597-022-01872-8)). This dataset has eleven different tests which can be grouped into the following broader categories:''')
    
    st.markdown(
          """
          - **Memory**: Forward Memory Span, Reverse Memory Span, Verbal List Learning, Delayed Verbal List Learning
          - **Information Processing Speed**: Digit Symbol Coding, Trail Making Part A, Trail Making Part B
          - **Reasoning**: Arithmetic Reasoning, Grammatical Reasoning
          - **Attention**: Divided Visual Attention
          - **Behavioral Restraint**: Go/No go
          """)
    
    st.markdown('''After you’ve set the weights to these five characteristics using the sliders, you can see which weights are assigned to each test (e.g. Forward Memory Span or Digit Symbol Coding) by ticking the checkbox beneath the sliders. ''')

col1, col2 = st.columns(2)

#Initialise slider values
list_values_A = (9, 10, 2, 1, 5)
list_values_B = (1, 2, 10, 9, 3)

selectionsA = {}
selectionsB = {}
results_dict_A = groups_dict
results_dict_B = groups_dict

with col1:
    st.subheader("Define target variable for model A ")
        
    if "slider_values_A" not in st.session_state:
        for count, value in enumerate(groups):
            selectionsA[value] = list_values_A[count]
        st.session_state["slider_values_A"] = selectionsA
    else:
        selectionsA = st.session_state["slider_values_A"]
            
    for i in groups:
        nameA = f"{i} importance, model A"
        value = selectionsA[i]
        slider = st.slider(nameA, min_value=0, max_value=10, value = value)
        selectionsA[i] = slider
        
    results_dict_A = {k: selectionsA.get(v, v) for k, v in results_dict_A.items()}
    total = sum(results_dict_A.values())
    for (key, u) in results_dict_A.items():
      if total != 0:
        w = (u/total)
        results_dict_A[key] = w

    if st.checkbox("Show target variable A weights per subtest", key="A"):
      for (key, u) in results_dict_A.items():
        txt = key.replace("_", " ")
        st.markdown("- " + txt + " : " + f":green[{str(round((u*100), 2))}]")
        
    
    st.session_state["slider_values_A"] = selectionsA


with col2:
    st.subheader("Define target variable for model B ")
        
    if "slider_values_B" not in st.session_state:
        for count, value in enumerate(groups):
            selectionsB[value] = list_values_B[count]
        st.session_state["slider_values_B"] = selectionsB
    else:
        selectionsB = st.session_state["slider_values_B"]
            
    for i in groups:
      nameB = f"{i} importance, model B"
      value = selectionsB[i]
      slider = st.slider(nameB, min_value=0, max_value=10, value = value)
      selectionsB[i] = slider
      
    results_dict_B = {k: selectionsB.get(v, v) for k, v in results_dict_B.items()}
    total = sum(results_dict_B.values())
    for (key, u) in results_dict_B.items():
      if total != 0:
        w = ((u/total))
        results_dict_B[key] = w

    if st.checkbox("Show target variable B weights per subtest", key = "B"):
      for (key, u) in results_dict_B.items():
        txt = key.replace("_", " ")
        st.markdown("- " + txt + " : " + f":green[{str(round((u*100), 2))}]")
      
    st.session_state["slider_values_B"] = selectionsB
          
if st.button("Assign labels and train your models", type = "primary", use_container_width = True):
  if 'complete_df' in st.session_state:
      del st.session_state['complete_df']
  if 'clean_df' in st.session_state:
      del st.session_state['clean_df']
  if 'cm_A' in st.session_state:
      del st.session_state['cm_A']
  if 'cm_B' in st.session_state:
      del st.session_state['cm_B']
  scoreA = pd.DataFrame()
  scoreB = pd.DataFrame()
  test1 = all(value == 0 for value in results_dict_A.values())
  test2 = all(value == 0 for value in results_dict_B.values())
  if test1 == True or test2 == True:
    st.error('Cannot train the models if you do not define the target variables. Make your selections for both models first!', icon="🚨")
  else:
    for (key, u) in results_dict_A.items():
      scoreA[df_keys_dict[key]] = u * dataframe[df_keys_dict[key]]
      scoresA = scoreA.sum(axis=1)
      dataframe['model_A_scores'] = scoresA
    for (key, u) in results_dict_B.items():
      scoreB[df_keys_dict[key]] = u * dataframe[df_keys_dict[key]]
      scoresB = scoreB.sum(axis=1)
      dataframe['model_B_scores'] = scoresB
    
    new_annotated = assign_labels_by_probabilities(dataframe, "model_A_scores", "Model_A_label", "Model_A_probabilities", quantile=0.85, num_samples=100)
    new_annotated = assign_labels_by_probabilities(new_annotated, "model_B_scores", "Model_B_label", "Model_B_probabilities", quantile=0.85, num_samples=100)
    new_annotated = new_annotated.reset_index()


    clean_data = drop_data(new_annotated)
    # specify the columns of interest
    selected_cols = ['Model_A_label', 'Model_B_label']
    
    # count the number of rows where all three selected columns have a value of 1
    num_rows_with_all_flags_1 = len(new_annotated[new_annotated[selected_cols].sum(axis=1) == len(selected_cols)])
    
    # print the result
    st.write(f"Shared candidates between your target variables: :green[{num_rows_with_all_flags_1}].")
    with st.spinner('Please wait... The models will be trained now.'):

      X_data, Y_data_A, Y_data_B = clean_data.iloc[:, :-2], clean_data.iloc[:, [-2]], clean_data.iloc[:, [-1]]
      X_data = X_data.drop(["index"], axis = 1)
      Y_data_B = Y_data_B.reset_index()
      X_train, X_test, y_train_A, y_test_A = train_test_split(X_data, Y_data_A, test_size=0.2)
      y_train_A = y_train_A.reset_index()
      y_test_A = y_test_A.reset_index()
      y_train_B = pd.merge(y_train_A,Y_data_B[['index', 'Model_B_label']],on='index', how='left')
      y_test_B = pd.merge(y_test_A,Y_data_B[['index', 'Model_B_label']],on='index', how='left')
      y_train_B = y_train_B.drop(labels='Model_A_label', axis = 1)
      y_test_B = y_test_B.drop(labels='Model_A_label', axis = 1)
      y_train_A = y_train_A.set_index("index")
      y_train_B = y_train_B.set_index("index")
      y_test_A = y_test_A.set_index("index")
      y_test_B = y_test_B.set_index("index")

      accuracy_A, precision_A, recall_A, X_full_A, cm_A, baseline_accuracy_A = train_and_predict("A", X_train, X_test, y_train_A, y_test_A)
      accuracy_B, precision_B, recall_B, X_full_B, cm_B, baseline_accuracy_B = train_and_predict("B", X_train, X_test, y_train_B, y_test_B)
      full = pd.merge(X_full_A,X_full_B[['index','Predicted_B', 'Prob_0_B', "Prob_1_B"]],on='index', how='left')
      complete = pd.merge(full,new_annotated[['index', 'age', 'gender', 'education level', 'country', 'Model_A_label', 'Model_B_label', 'model_A_scores', 'model_B_scores']],on='index', how='left')
      complete=complete.replace({"education level": education_dict})
      complete = complete.rename(columns={"index": "Candidate ID"})
      
      if 'complete_df' not in st.session_state:
        st.session_state['complete_df'] = complete
      if 'clean_df' not in st.session_state:
        st.session_state['clean_df'] = clean_data
      if 'cm_A' not in st.session_state:
        st.session_state['cm_A'] = cm_A
      if 'cm_B' not in st.session_state:
        st.session_state['cm_B'] = cm_B

    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))
    with row1_1:
      st.write(f"Model A accuracy: :green[{baseline_accuracy_A}].")
    with row1_2:
      st.write(f"Model B accuracy: :green[{baseline_accuracy_B}].")

    st.success('''Success! You have defined the target variables and trained your models. Head to "See visualisations" in the sidebar.''')