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
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from utils import create_confusion_matrix_heatmap, plot_conf_rates, assign_labels_by_probabilities, drop_data_exp_2, train_and_predict, plot_data, run_PCA

### DICTIONARIES AND CONSTANTS ###
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
          'education_level' : colours_education,
          'country' : colours_country,
          'age' : 'indigo'
        }

model_dict = {
          'Model A' : 'Model_A_label',
          'Model B' : 'Model_B_label'
        }

pred_dict = {
          'Model A' : 'Predicted_A',
          'Model B' : 'Predicted_B'
        }

prob_dict = {
          'Model A' : 'Prob_1_A',
          'Model B' : 'Prob_1_B'
        }

color_dict = {0: 'lightseagreen',
          1: 'darkcyan',
          2: 'darkslategrey',
          3 : 'royalblue',
          4 : 'deepskyblue',
          5: 'midnightblue',
          6 : 'purple',
          7: 'dodgerblue',
          8 : 'slategray',
          9: 'blue',
          10 : 'slateblue',
          11 : 'mediumblue'}

protected_chars = ['education_level', 'country', 'age', 'gender']
models = ['Model A', 'Model B']

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

### IMPORT DATA FILES ###
dataframe = pd.read_csv('data/dataframe.csv')
dataframe = dataframe.drop(["Unnamed: 0"], axis = 1)

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')

### CREATE THE "ABOUT" PAGE ###
def about():
  st.dataframe(data = dataframe)
  #Set title
  st.title('EquiVar - a new paradigm in hiring by machine')
  # Set columns
  c1, c2, c3, c4 = st.columns(4)

  #Create info boxes for authors, links and GitHub
  with c1:
      st.info('**Data Scientist: [Dalia Sara Gala](https://twitter.com/dalia_science)**')
  with c2:
      st.info('**Philosophy Lead: [Milo Phillips-Brown](https://www.milopb.com/)**')
  with c3:
      st.info('**Accenture team: [@The Dock](https://www.accenture.com/gb-en/services/about/innovation-hub-the-dock)**')
  with c4:
      st.info('**GitHub: [Hiring-model](https://github.com/DaliaGala/Hiring-model)**')

  #Project description
  st.subheader('Motivation')
  st.write(
      """
            Hiring applications often require for recruiters and employers to sift through a large volume of applications. With the progress in the development of more sophisticated machine learning algorithms, companies are becoming more reliant on such systems to produce lists of candidates who could be potential hires. These algorithms can, for example, rely on keyword matching for candidate CVs and job requirements. Recently, a new type of algorithms are becoming more extensively used, which try to deploy games, riddles and challenges to capture some of the candidates' qualities. Through these, companies can assess the candidates' capacities by comparing their results to those of successful incumbent successful employees.

A concern which emerges around the concept of a "successful employee" in the hiring context is the following: how does one define what a successful employee is? There might by **multiple ways** of defining this. As [Barocas and Selbst (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899) outline: _“Good” must be defined in ways that correspond to measurable outcomes: relatively higher sales, shorter production time, or longer tenure, for example._ The model creators must then _translate some amorphous problem into a question that can be expressed in more formal terms that computers can parse_, which is often an ambiguous task. Therefore, the definition of a "good employee" or the way in which the **target variable** is specified, hugely impacts the model outcomes.

As such, any social application of such technologies carries with it risks related to fairness and bias. Inevitably, machine learning models are but mere generalisations and, in issues as complex as hiring, their use can lead to disparate outcomes for protected groups and minorities. Moreover, some of these algorithms can inadvertently become trained to select candidates based on a feature directly correlated with some of their protected characteristics, called "selection by proxy". Such selection is illegal in many countries in the world, where equal employment opportunity laws prevent the employers from making their decisions based on sex, race, religion or belief, ethnic or national origin, disability, age, citizenship, marital, domestic or civil partnership status, sexual orientation, gender identity, pregnancy or related condition (including breastfeeding) or any other basis as protected by applicable law. 

Usually, to address these issues, constraints are put on the models which are being trained and deployed. These could take a form of, for example, mandating that the proportions of candidates selected by the model be correlated with the proportions of the input populations, etc. However, such measures often result in **accuracy versus fairness** problems. This describes a situation in which achieving better fairness by some specified fairness metric results in lower accuracy of the model.

To tackle these problems, we have designed a new paradigm for hiring by the machine, which we called Equi Var, for Equivalent Target Variable. This work was carried out to achieve the following goals:

1. Propose a new paradigm of training hiring algorithms which select "successful candidates" based on equivalent, yet distinct, target variable definitions.

2. Demonstrate that such algorithms, though nominally equivalent, can produce differing outcomes with respect to protected characteristics of assessed candidates.

3. Present how our paradigm can be used to mitigate the accuracy versus fairness problem.

      """
  )
  st.subheader('Methodology')
  st.write(
      """
Due to the nature of our work and the data related to it, it proved challenging to find an open-source labelled dataset pertaining to hiring. Drawing on the examples of gamified algorithms, we explored datasets related to cognitive challenges and came across the [**NCPT dataset**](https://www.nature.com/articles/s41597-022-01872-8). This dataset contains scores from adults who completed the NeuroCognitive Performance Test (NCPT; Lumos Labs, Inc.). This is a self-administered cognitive test which can be performed by adults who sign up for Lumosity training program, aimed at improving memory, attention, flexibility and problem solving of participants. The NCPT is offered to Lumosity participants before they start training to assess their initial abilities. 

This dataset contained between 5 and 11 subtests per data battery. Since these subtests were aimed at examining qualities such as working memory, visual attention, and abstract reasoning, this data struck us as similar enough to what could be collected in gamified assessments. Moreover, this dataset includes basic demographic information from each participant, which was necessary for the realisation of our goals.

This dataset was, naturally, not labelled in the context of a hiring algorithm. This presented us with a perfect opportunity to utilise it to demonstrate how the varying importance of these subtests - and therefore the characteristics which they examine - affects the definition
of the target variable. We allow you to try to decide which characteristics you think should count for an employee to be deemed "successful". You can create two toy hiring models and observe how your choices impact
the target demographics. For more details, and to create your models, head to the "Define target variable" section in the sidebar. For more details, read our paper.""")

### CREATE THE SIDEBAR ###
st.sidebar.markdown("Contents")

# Add selection to the sidebar
with st.sidebar:
    add_radio = st.radio(
        "Choose option: explore input data",
        ("About", "Define target variable", "Visualise your models")
    )

### CREATE THE "DATA VISUALISATION" PAGE ###

def model_scores(key1):

  # Create a selectbox to choose a protected characteristic to explore
  plot_radio = st.radio('Characteristic to explore', characteristic_dict.keys(), horizontal=True)
  row2_space1, row2_1, row2_space2 = st.columns((0.1, 5, 0.1))

  if key1 not in st.session_state:
    st.error('Cannot train the model if you do not define the target variable. Make your selections first!', icon="🚨")
  else:
    dataframe = st.session_state[key1]

    with row2_1:
      data = dataframe[["model_A_scores", "model_B_scores", plot_radio]]

      if plot_radio == "age":
        selectbox_Mod = st.selectbox('Choose model', ("Model A", "Model B"))
        if selectbox_Mod == "Model A":
          fig = px.scatter(data, x=data['age'], y="model_A_scores", trendline="ols")
          st.write(fig)
        else:
          fig = px.scatter(data, x=data['age'], y="model_B_scores", trendline="ols")
          st.write(fig)
              
      else:
        
        fig = go.Figure(layout=go.Layout(height=700, width=900))

        fig.add_trace(go.Box(
            y = data["model_A_scores"],
            x = data[plot_radio],
            name = 'Model A Scores',
            marker_color = '#3D9970'
        ))
        fig.add_trace(go.Box(
            y = data["model_B_scores"],
            x = data[plot_radio],
            name = 'Model B Scores',
            marker_color='#FF4136'
        ))

        fig.update_layout(
            yaxis_title='model scores',
            boxmode='group' # group together boxes of the different traces for each value of x
        )
        st.write(fig)

def data_vis(key1, key2):
  add_radio = st.radio("Choose option: explore train and test data", ("PCA", "Component loadings - PCA", "Data charactertistics"), horizontal=True)

  if key2 not in st.session_state:
    st.error('Cannot train the model if you do not define the target variable. Make your selections first!', icon="🚨")
  else:
    dataframe_PCA = st.session_state[key2]
    full_df = st.session_state[key1]

    if add_radio == "PCA":
      selectbox = st.selectbox('Model to explore', ('Model A', 'Model B'))

      if selectbox == 'Model A':
        pcaA, dfA, labelsA, coeffA, componentsA = run_PCA(dataframe_PCA, 'Model_B_label', 'Model_A_label', 3)

        total_var = pcaA.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            componentsA, x=0, y=1, z=2, color=dfA['target'],
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.update_traces(marker_size = 5)
        st.plotly_chart(fig)
      else:
        pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 3)
        total_var = pcaB.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            componentsB, x=0, y=1, z=2, color=dfB['target'],
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.update_traces(marker_size = 5)
        st.plotly_chart(fig)

    if add_radio == "Component loadings - PCA":
      pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 2)
      loadings = pcaB.components_.T * np.sqrt(pcaB.explained_variance_)

      fig = px.scatter(componentsB, x=0, y=1)

      for i, feature in enumerate(labelsB):
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
      st.plotly_chart(fig)

  if add_radio == "Data charactertistics":
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
    )
    if key2 not in st.session_state:
      pass
    else:
      dataframe_PCA = st.session_state[key2]
      full_df = st.session_state[key1]

      # Plot training data
      with row1_1:
        st.subheader("Training data")

        # Select train data
        train = full_df.loc[full_df["Predicted_A"] == "train"]

        # Create a selectbox to choose a protected characteristic to explore
        selectbox = st.selectbox('Train characteristic', characteristic_dict.keys())

        # Use function plot_data to plot selected data
        plot_data(train, selectbox, characteristic_dict[selectbox])

      # Plot test data

      with row1_2:
        st.subheader("Test data")

        # Select test data
        test = full_df.loc[full_df["Predicted_A"] != "train"]

        # Create a selectbox to choose a protected characteristic to explore
        selectbox = st.selectbox('Test characteristic', characteristic_dict.keys())

        # Use function plot_data to plot selected data
        plot_data(test, selectbox, characteristic_dict[selectbox])

def model_out(key1):
  add_radio = st.radio ("Choose what to explore", ("Demographics", "See the output data", "Venn Diagram"), horizontal=True)
  if key1 not in st.session_state:
    pass
  else:
    full_df = st.session_state[key1]

    if add_radio == "Demographics":
      # Create a selectbox to choose a protected characteristic to explore
      selectbox = st.selectbox('Characteristic to explore', characteristic_dict.keys())
      row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))
      with row1_1:
        st.subheader("Model A")

        # Select test data
        data = full_df.loc[full_df['Predicted_A'] == 1]

        # Use function plot_data to plot selected data
        plot_data(data, selectbox, characteristic_dict[selectbox])

      with row1_2:
        st.subheader("Model B")

        # Select test data
        data = full_df.loc[full_df['Predicted_B'] == 1]

        # Use function plot_data to plot selected data
        plot_data(data, selectbox, characteristic_dict[selectbox])

    if add_radio == "See the output data":
      selectbox_M = st.selectbox('Choose a model', pred_dict.keys())

      # Select data
      data = full_df.loc[full_df[pred_dict[selectbox_M]] == 1]
      data = data.sort_values(by = prob_dict[selectbox_M], ascending = False)
      data = data[['Candidate ID','Prob_1_A', 'Prob_1_B', 'Predicted_A', 'Predicted_B']]
      data.index = np.arange(1, len(data) + 1)

      # # CSS to inject contained in a string
      # hide_table_row_index = """
      #         <style>
      #         thead tr th:first-child {display:none}
      #         tbody th {display:none}
      #         </style>
      #         """

      # # Inject CSS with Markdown
      # st.markdown(hide_table_row_index, unsafe_allow_html=True)

      st.table(data.style.background_gradient(subset = ['Prob_1_A', 'Prob_1_B'], axis=0, vmin=0.40).highlight_max(color = '#FFCD9B', subset = ['Predicted_A', 'Predicted_B'], axis=0))
      #text_gradient(cmap = 'winter', axis=0, subset = ['Prob_1_A', 'Prob_1_B', 'Prob_1_C'], vmin=0.70))
      #st.table(data.style.highlight_max(subset = ['Prob_1_A', 'Prob_1_B', 'Prob_1_C'], axis=0))

    if add_radio == "Venn Diagram":
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
        st.markdown('This Venn Diagram presents the number of candidates which were selected by both models.')

def model_vis(key1):
  if key1 not in st.session_state:
    pass
  else:
    full_df = st.session_state[key1]

    # Create a selectbox to choose a protected characteristic to explore
    selectbox_Lab = st.selectbox('Label to explore', ('Positive labels', 'Negative labels'))

    # Create a selectbox to choose a protected characteristic to explore
    selectbox_Char = st.selectbox('Characteristic to explore', characteristic_dict.keys())

    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))

    with row2_1:
      st.subheader("Model A")

      # Select test data
      if selectbox_Lab == 'Positive labels':
        data = full_df.loc[full_df['Model_A_label'] == 1]
      else:
        data = full_df.loc[full_df['Model_A_label'] == 0]
      
      # Use function plot_data to plot selected data
      plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])


    with row2_2:
      st.subheader("Model B")

      # Select test data
      if selectbox_Lab == 'Positive labels':
        data = full_df.loc[full_df['Model_B_label'] == 1]
      else:
        data = full_df.loc[full_df['Model_B_label'] == 0]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])

def mod_prop(key3, key4):
  if key3 not in st.session_state:
    pass
  else:
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 3, 0.1, 3, 0.1))

    with row1_1:
      st.header("Model A confusion matrix")
      create_confusion_matrix_heatmap(st.session_state[key3])
      plot_conf_rates(st.session_state[key4])

    with row1_2:
      st.header("Model B confusion matrix")
      create_confusion_matrix_heatmap(st.session_state[key4])
      plot_conf_rates(st.session_state[key3])
        
def data_plot(key1, key2, key3, key4):
  st.title('''Compare the models you trained''')
  page_radio = st.radio('What would you like to see?', ("Model outcomes", "Input scores", "Principal Component Analysis", "Label distribution", "Confusion matrices"), horizontal=True)

  if page_radio == "Input scores":
    model_scores(key1)
  if page_radio == "Principal Component Analysis":
    data_vis(key1, key2)
  if page_radio == "Model outcomes":
    model_out(key1)
  if page_radio == "Label distribution":
    model_vis(key1)
  if page_radio == "Confusion matrices":
    mod_prop(key3, key4)

### CREATE THE "TARGET VARIABLE DEFINITION" PAGE ###
def define_target_variable():

  st.title('Selection of target variable')
  expander = st.expander("See explanation")
  
  expander.write('''On this page, you can participate in the process of defining the target variable for a hiring model. The target variable is 
  the variable, or a changeable quality, whose value is defined and predicted by other variables. In the case of a hiring model, the target variable
  usually is: who should be hired? Or: who should be interviewed? Therefore, the target variable will be a group of people. Once this group is defined 
  in some way, we can show the model some examples of "features" of these people. In the case of the gamified hiring assessments, the group is defined
  by someone in a company, possibly a hiring manager, who will select a group of "top performers". Then, these employees will play the cognitive games,
  thus generating the features. The features will be their performance in those games. Finally, when new applicants apply for jobs, they will be asked
  to play the same games. If the applicants' results are similar to those achieved by the top employees, the model might select them for interview/hiring.
  If not, the model will not select them.
  
  Therefore, it is easy to imagine and understand that the way in which the hypothetical hiring manager selects the group of top employees will have a 
  large effect on who the model learns to select. One manager might favour attentiveness and numerical skills more, while another might be more focused
  on interpersonal skills and having good memory. When these two groups of people play the games, their results, and what the model learns to detect, 
  will differ. Not only that; the demographics selected by these models will most likely be very different too.
  
  Here, you can define the target variable for two hiring models yourself, and see the effect that these varying definitions have on the target demographics 
  of your models. Once you are finished setting the slider values, press "Assign values and train your models" button, and head to "Visualise your models".''')

  col1, col2 = st.columns(2)

  selectionsA = {}
  selectionsB = {}

  groups = ["attention", "reasoning", "memory", "behavioural restraint", "information processing speed"]
  results_dict_A = groups_dict
  results_dict_B = groups_dict

  with col1:
    st.write("Define target variable for model A ")
    for i in groups:
      name = f"{i} importance, model A"
      slider = st.slider(name, min_value=0, max_value=10)
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

  with col2:
    st.write("Define target variable for model B ")
    for i in groups:
      name = f"{i} importance, model B"
      slider = st.slider(name, min_value=0, max_value=10)
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
            
  if st.button("Assign labels and train your models"):
    scoreA = pd.DataFrame()
    scoreB = pd.DataFrame()
    test = all(value == 0 for value in results_dict_A.values()) and all(value == 0 for value in results_dict_B.values())
    if test == True:
      st.error('Cannot train the model if you do not define the target variable. Make your selections first!', icon="🚨")
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


      clean_data = drop_data_exp_2(new_annotated)
      # specify the columns of interest
      selected_cols = ['Model_A_label', 'Model_B_label']
      
      # count the number of rows where all three selected columns have a value of 1
      num_rows_with_all_flags_1 = len(new_annotated[new_annotated[selected_cols].sum(axis=1) == len(selected_cols)])
      
      # print the result
      st.write("Shared candidates between your target variables:", num_rows_with_all_flags_1)
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

      accuracy_A, precision_A, recall_A, X_full_A, cm_A = train_and_predict("A", X_train, X_test, y_train_A, y_test_A)
      accuracy_B, precision_B, recall_B, X_full_B, cm_B = train_and_predict("B", X_train, X_test, y_train_B, y_test_B)
      full = pd.merge(X_full_A,X_full_B[['index','Predicted_B', 'Prob_0_B', "Prob_1_B"]],on='index', how='left')
      complete = pd.merge(full,new_annotated[['index', 'age', 'gender', 'education_level', 'country', 'Model_A_label', 'Model_B_label', 'model_A_scores', 'model_B_scores']],on='index', how='left')
      complete=complete.replace({"education_level": education_dict})
      complete = complete.rename(columns={"index": "Candidate ID"})
      
      if 'complete_df' not in st.session_state:
        st.session_state['complete_df'] = complete
      if 'clean_df' not in st.session_state:
        st.session_state['clean_df'] = clean_data
      if 'cm_A' not in st.session_state:
        st.session_state['cm_A'] = cm_A
      if 'cm_B' not in st.session_state:
        st.session_state['cm_B'] = cm_B

      st.markdown('''You are done defining the target variable. Click "Visualise your models" in the sidebar to see the results!''')


### ASSIGN WHAT TO DISPLAY ACCORDING TO THE SIDEBAR SELECTION ###
  
if add_radio == 'About':
  about()
if add_radio == "Define target variable":
  define_target_variable()
if add_radio == "Visualise your models":
  data_plot('complete_df', 'clean_df', 'cm_A', 'cm_B')