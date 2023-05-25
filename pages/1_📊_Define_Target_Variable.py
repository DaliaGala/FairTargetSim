#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:16:50 2023

@author: daliagala
"""

### IMPORT LIBRARIES ###
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from utils import assign_labels_by_probabilities, drop_data, train_and_predict


### IMPORT DATA FILES ###
dataframe = pd.read_csv('./data/dataframe.csv')
dataframe = dataframe.drop(["Unnamed: 0"], axis = 1)
dataframe = dataframe.rename(columns={"education_level": "education level"})

### DICTIONARIES AND CONSTANTS###
groups = ["attention", "reasoning", "memory", "behavioural restraint", "information processing speed"]

groups_dict = {
          'divided_visual_attention' : 'attention',
          'forward_memory_span': 'memory',
          'arithmetic_problem_solving' : 'reasoning',
          'logical_reasoning' : 'reasoning',
          'adaptive_behaviour_response_inhibition' : 'behavioural restraint',
          'reverse_memory_span' : 'memory',
          'episodic_verbal_learning' : 'memory',
          'delayed_recall' : 'memory',
          'abstract_symbol_processing_speed' : 'information processing speed',
          'numerical_info_processing_speed' : 'information processing speed',
          'numerical_and_lexical_info_processing_speed' : 'information processing speed'
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



### CREATE THE "TARGET VARIABLE DEFINITION" PAGE ###
st.title('Target variable definition')

with st.expander("See explanation"):
    st.markdown('''On this page, you can participate in the process of defining the target variable for a hiring model. The :green[target variable is 
    the variable, or a changeable quality, whose value is defined and predicted by other variables]. In the case of a hiring model, the target variable
    usually is: who should be hired? Or: who should be interviewed? Therefore, the target variable will be **a group of people**. Those to be selected will be
    labelled "1", and those who are not to be selected, will be labelled "0". This is an example of a **classification task** in machine learning. Once these 
    groups are defined in some way, we can show the model some examples of "features" which determine who should belong to which group.''')
    
    st.markdown('''In a real-world scenario,the target variable would be defined by a hiring manager based on their **implicit assessment** of current employees. 
    The hiring manager would decide who she thinks performs well, and who does not, in her company. She would then ask those employees to play 
    the cognitive games for the purpose of generating features based on which the hiring algorithm can be trained. The algorithm would know who was selected to 
    have the label "1", and who was selected to have the label "0" because that was what the hiring manager decided. But if another hiring manager was making this 
    decision on a different occasion, he might have a **different baseline** with respect to which he would select those employees. Both hiring managers therefore 
    :green[implicitly assign different importance to different characteristics of incumbent employees:] how good they are at presentations, how quickly they complete tasks, 
    how good their memory is and how quickly they learn new things, or similar. Finally, when new applicants apply for a job in this company, they will also be 
    asked to play the same games. If the applicants' results are similar to those achieved by the top employees, the model might select them for interview or hiring.
    They will be labelled "1" by the model. If not, the model will not select them. They will be labelled "0".''')
    
    st.markdown('''Therefore, it is easy to imagine and understand that the way in which the hypothetical hiring manager selects the group of top employees will have a 
    :green[large effect on who the model learns to select]. One manager might favour attentiveness and numerical skills more, while another might be more focused
    on interpersonal skills and having good memory. When these two groups of people play the games, their results, and what the model learns to detect, 
    will differ. Not only that; :green[the demographics selected by these models will most likely be very different, too.]''')

    st.markdown('''Here, :green[**you can define the target variable**] for two hiring models yourself in an **explicit** manner, and see the effect that these varying definitions have on the target demographics 
    of your models. Rather than weighing different features of hypothetical employees in your head, and deciding "1", or "0", you will do so numerically. To do that, we preprocessed and normalised
    data from battery 26 of the [**NCPT dataset**](https://www.nature.com/articles/s41597-022-01872-8) which is a dataset with results of cognitive performance tests. We selected this battery because
    it had the most subtests from all batteries in the NCPT dataset, 11 subtests, and therefore, most available features to use. We treat the results of each of the subtests as features for a hiring model,
    and we group them into 5 categories based on what the subtests measure:''')

    c1, c2, c3, c4, c5 = st.columns((2.5,3,2,1,2))
    with c1:
      st.markdown(
          """
          **Memory**:
          - forward memory span
          - reverse memory span
          - episodic verbal learning
          - delayed recall
          """)
    with c2:
      st.markdown(
          """
          **Information processing speed**:
          - abstract symbol processing speed
          - numerical info processing speed
          - numerical and lexical info processing speed
          """)
    with c3:
      st.markdown(
          """
          **Reasoning**:
          - arithmetic problem solving 
          - logical reasoning
          """)
    with c4:
      st.markdown(
          """
          **Attention**:
          - divided visual attention
          """)
    with c5:
      st.markdown(
          """
          **Behavioural restraint**:
          - adaptive behaviour response inhibition
          """)
    
    st.markdown('''Below, you can see sliders corresponding to these 5 categories, and thus, the characteristics of hypothetical employees. You can assign weights to these features by changing the slider values, and these
    weights will be translated into the importance of these features when assigning a score to the hypothetical employee. Once you are finished setting the slider values, you can see what % values you assigned to 
    each subtest by ticking the checkbox below the sliders. When you are finished with your selections, press "Assign values and train your models" button. Scores will be assigned to each person in the dataset 
    based on your selection. These scores will be different for models A and B if you selected varying slider values for the characteristics which are used as features in these models. In the top 15% of candidates
    based on each score for each model A and B, 100 candidates will be labelled "1", and the rest of the candidates in the dataset will have the label "0". The candidates will be labelled "1" with increasing probability 
    of being selected the higher their score was. This is done to emulate what a real-world decision might look like, and to introduce noise to the model and prevent it from being 100% accurate.''')

col1, col2 = st.columns(2)

selectionsA = {}
selectionsB = {}
results_dict_A = groups_dict
results_dict_B = groups_dict

with col1:
    st.subheader("Define target variable for model A ")
    for i in groups:
        selectionsA[i] = 0
        
    if "slider_values_A" not in st.session_state:
        st.session_state["slider_values_A"] = selectionsA
    else:
        selectionsA = st.session_state["slider_values_A"]
            
    for i in groups:
        nameA = f"{i} importance, model A"
        value = selectionsA[i]
        slider = st.slider(nameA, min_value=0, max_value=10, value = value)
        selectionsA[i] = slider
        
    
    st.session_state["slider_values_A"] = selectionsA
  
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

with col2:
    st.subheader("Define target variable for model B ")
    for i in groups:
        selectionsB[i] = 0
        
    if "slider_values_B" not in st.session_state:
        st.session_state["slider_values_B"] = selectionsB
    else:
        selectionsB = st.session_state["slider_values_B"]
            
    for i in groups:
      nameB = f"{i} importance, model B"
      value = selectionsB[i]
      slider = st.slider(nameB, min_value=0, max_value=10, value = value)
      selectionsB[i] = slider
      
    st.session_state["slider_values_B"] = selectionsB
    
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
            
if st.button("Assign labels and train your models", type = "primary", use_container_width = True):
    scoreA = pd.DataFrame()
    scoreB = pd.DataFrame()
    test1 = all(value == 0 for value in results_dict_A.values())
    test2 = all(value == 0 for value in results_dict_B.values())
    if test1 == True or test2 == True:
      st.error('Cannot train the models if you do not define the target variables. Make your selections for both models first!', icon="🚨")
    else:
      for (key, u) in results_dict_A.items():
        scoreA[key] = u * dataframe[key]
        scoresA = scoreA.sum(axis=1)
        dataframe['model_A_scores'] = scoresA
      for (key, u) in results_dict_B.items():
        scoreB[key] = u * dataframe[key]
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

      st.success('''Success! You have defined the target variables and trained your models. Click "Visualise your models" in the sidebar to explore.''')