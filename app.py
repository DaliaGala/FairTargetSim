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
from utils import create_confusion_matrix_heatmap, plot_conf_rates, plot_data, create_selectbox, run_PCA, plot_no_loadings, plot_loadings

### IMPORT DATA FILES ###
df = pd.read_csv('./data/dataframe.csv')
cm_A = pd.read_csv('./data/cm_A.csv')
cm_A = cm_A.drop(["Unnamed: 0.1", "Unnamed: 0"], axis = 1)
cm_B = pd.read_csv('./data/cm_B.csv')
cm_B = cm_B.drop(["Unnamed: 0.1", "Unnamed: 0"], axis = 1)
cm_C= pd.read_csv('./data/cm_C.csv')
cm_C = cm_C.drop(["Unnamed: 0.1", "Unnamed: 0"], axis = 1)
PCA_df = pd.read_csv('./data/dataframe_PCA.csv')
PCA_df = PCA_df.drop(["Unnamed: 0"], axis = 1)
model_scores_df = pd.read_csv('./data/model_scores.csv')

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')


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
          'education_level' : colours_education,
          'gender' : colours_gender,
          'country' : colours_country,
          'age' : 'indigo'
        }

model_dict = {
          'Model A' : 'Model_A_label',
          'Model B' : 'Model_B_label',
          'Model C' : 'Model_C_label'
        }

pred_dict = {
          'Model A' : 'Predicted_A',
          'Model B' : 'Predicted_B',
          'Model C' : 'Predicted_C'
        }

prob_dict = {
          'Model A' : 'Prob_1_A',
          'Model B' : 'Prob_1_B',
          'Model C' : 'Prob_1_C'
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
models = ['Model A', 'Model B', 'Model C']

### CREATE THE "ABOUT" PAGE ###

def about():
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

This dataset was, naturally, not labelled in the context of a hiring algorithm. We were therefore presented with the challenge of applying class labels in order to be able to produce models. We applied binary class labels. We wanted to label participants as having a label "0", which would mean that this participant would not be selected for the job position or interview, and a label of "1", which would mean that the participant would be selected. Moreover, we wanted to have more than 1 way of deciding how these labels will be administered. We prepared 3 datasets, which we subsequently refer to as A, B and C, each with different distribution of positive and negative labels. This distribution was achieved by calculating different weighted averages of the subtest results for each test. Below, we describe the process by which we assigned these labels:

1. We selected the data battery from the NCPT dataset with the largest number of subtests to maximise the number of available features. This was battery 26, with 11 features.

2. We then created subgroups of subtests from this battery. In this process, we tried to mimic the logic of what 3 different hiring managers might consider important to their selection of candidates. We assumed the 3 hiring managers might agree on some characteristics of the candidates and deem them necessary for each candidate. We assumed that some tests will be the ones which measure these characteristics, and that, by proxy, all 3 managers will agree that selected candidates will have to have high scores in these tests. These tests (n=5) were Grammatical Reasoning test, Trail Making A, Trail Making B, Forward memory test, Reverse memory test. Each test in this group was assigned a weight of 6.4% in the weighted average. The total weight of these tests in the weighted average score for each model A, B and C was therefore 32%.

3. We then selected 3 groups of 2 tests each which were to be the distinguishing tests between the models. We selected these groups by the logic that while all 3 hiring managers might agree on some of the characteristics of the candidates, they might disagree on others. In this hypothetical scenario, we therefore reasoned that:

- manager A thinks that quantitative skills and focus are important; in this model, Arithmetic reasoning test and Divided visual attention tests are given the weight of 25% each, for a total of 50%; the remaining 4 tests have a weight of 4.5%

- manager A thinks that memory and recall are most important; in this model, Verbal list learning and Delayed verbal lists learning tests are given the weight of 25% each, for a total of 50%; the remaining 4 tests have a weight of 4.5%

- manager A thinks that behavioural restraint and quick information processing skills are key; in this model, the response inhibition test (Go/no go) and the information processing speed test (Digit symbol coding) are given the weight of 25% each, for a total of 50%; the remaining 4 tests have a weight of 4.5%""")

  # Set columns
  c1, c2 = st.columns(2)

  with c1:
    st.write(
      """ 4. Based on these weighted scores, for each model we remove the bottom 85% of scores and assign a linearly increasing likelihood of being selected when score increases to the top 15% (see figure).""")
  with c2:
    st.image('./images/model_image.PNG', use_column_width = True)



### CREATE THE SIDEBAR ###

st.sidebar.markdown("Contents")

# Add selection to the sidebar
with st.sidebar:
    add_radio = st.radio(
        "Choose option: explore input data",
        ("About", "Model scores", "Data exploration", "Model labels distribution", "Model confusion matrices", "Model outcomes")
    )

### CREATE THE "DATA VISUALISATION" PAGE ###

def data_plot():
  st.subheader('Visualising the demographics data vs score per model')
  st.write("""In this section we look at the input data for each model. What is the relationship between the scores which were calculated for each candidate based on selection criteria when compared between different protected groups? An interesting observation which was also made by the authors of the [**NCPT paper**](https://www.nature.com/articles/s41597-022-01872-8) was that the scores per individual game, but also the overall scores, negatively correlate with age. Therefore, it appears that gamified assessments have the potential to disadvantage older applicants from the get-go, if we assume that the conclusions from the NCPT dataset extend to other games which might be played in gamified hiring assessments.""")

  # Create a selectbox to choose a protected characteristic to explore
  selectbox_Char = create_selectbox('Characteristic to explore', characteristic_dict.keys())

  row2_space1, row2_1, row2_space2 = st.columns((0.1, 5, 0.1))

  with row2_1:
    # st.subheader("Model A")
    # data = model_scores_df[["model_A_scores", selectbox_Char]]
    # fig = px.box(data, x = selectbox_Char, y="model_A_scores", color=selectbox_Char)
    # st.plotly_chart(fig, use_container_width=True)
    data = model_scores_df[["model_A_scores", "model_B_scores", "model_C_scores", selectbox_Char]]

    if selectbox_Char == "age":
      selectbox_Mod = create_selectbox('Choose model', ("Model A", "Model B", "Model C"))
      if selectbox_Mod == "Model A":
        fig = px.scatter(data, x=data['age'], y="model_A_scores", trendline="ols")
        st.write(fig)
      elif selectbox_Mod == "Model B":
        fig = px.scatter(data, x=data['age'], y="model_B_scores", trendline="ols")
        st.write(fig)
      else:
        fig = px.scatter(data, x=data['age'], y="model_C_scores", trendline="ols")
        st.write(fig)
            
    else:
      
      fig = go.Figure(layout=go.Layout(height=700, width=900))

      fig.add_trace(go.Box(
          y = data["model_A_scores"],
          x = data[selectbox_Char],
          name = 'Model A Scores',
          marker_color = '#3D9970'
      ))
      fig.add_trace(go.Box(
          y = data["model_B_scores"],
          x = data[selectbox_Char],
          name = 'Model B Scores',
          marker_color='#FF4136'
      ))
      fig.add_trace(go.Box(
          y = data["model_C_scores"],
          x = data[selectbox_Char],
          name = 'Model C Scores',
          marker_color='#FF851B'
      ))

      fig.update_layout(
          yaxis_title='model scores',
          boxmode='group' # group together boxes of the different traces for each value of x
      )
      st.write(fig)

### CREATE THE "DATA VISUALISATION" PAGE ###

def data_vis():
  st.subheader('Visualising the demographics of the training and test data')
  tab1, tab2, tab3 = st.tabs(["PCA", "Components loadings - PCA", "Data charactertistics"])

  with tab1:
    row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space4= st.columns((0.1, 2, 0.1, 2, 0.1, 2, 0.1))
    with row1_1:
      st.subheader("Model A PCA")
      pcaA, dfA, labelsA, coeffA, componentsA = run_PCA(PCA_df, 'Model_B_label', 'Model_C_label', 'Model_A_label', 2)
      plot_no_loadings(dfA)

    with row1_2:
      st.subheader("Model B PCA")
      pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(PCA_df, 'Model_A_label', 'Model_C_label', 'Model_B_label', 2)
      plot_no_loadings(dfB)
    
    with row1_3:
      st.subheader("Model C PCA")
      pcaC, dfC, labelsC, coeffC, componentsC = run_PCA(PCA_df, 'Model_A_label', 'Model_B_label', 'Model_C_label', 2)
      plot_no_loadings(dfC)

  with tab2:
    row1_space1, row1_1, row1_space2 = st.columns((1, 4, 1))

    with row1_1:
      st.subheader("PCA - component loadings")
      pcaC, dfC, labelsC, coeffC, componentsC = run_PCA(PCA_df, 'Model_A_label', 'Model_B_label', 'Model_C_label', 2)
      fig = plt.figure(figsize = (6,6))
      ax = fig.add_subplot(1,1,1) 
      ax.set_xlabel('Principal Component 1', fontsize = 12)
      ax.set_ylabel('Principal Component 2', fontsize = 12)
      ax.set_title('2 component PCA', fontsize = 12)
      ax.tick_params(axis='both', which='major', labelsize=8)
      ax.tick_params(axis='both', which='minor', labelsize=8)
      plt.scatter(dfC['principal component 1'], dfC['principal component 2'], c = 'orange', s = 10, alpha = 0.6)
      for i in range(coeffC.shape[0]):
        plt.arrow(0, 0, coeffC[i,0], coeffC[i,1], width = 0.01, color=color_dict[i])
        texts = [plt.text(coeffC[i, 0], coeffC[i, 1], labelsC[i], ha='center', va='center', fontsize = 'small', fontvariant = 'small-caps', fontweight = 'bold', color=color_dict[i]) for i in range(coeffC.shape[0])]
        adjust_text(texts)
      plt.grid()
      st.pyplot(fig)


  with tab3:
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
    )

    # Plot training data
    with row1_1:
      st.subheader("Training data")

      # Select train data
      train = df.loc[df["Predicted_A"] == "train"]

      # Create a selectbox to choose a protected characteristic to explore
      selectbox = create_selectbox('Train characteristic', characteristic_dict.keys())

      # Use function plot_data to plot selected data
      plot_data(train, selectbox, characteristic_dict[selectbox])

    # Plot test data

    with row1_2:
      st.subheader("Test data")

      # Select test data
      test = df.loc[df["Predicted_A"] != "train"]

      # Create a selectbox to choose a protected characteristic to explore
      selectbox = create_selectbox('Test characteristic', characteristic_dict.keys())

      # Use function plot_data to plot selected data
      plot_data(test, selectbox, characteristic_dict[selectbox])


### CREATE THE "MODEL DATA VISUALISATION" PAGE ###
      
def model_vis():
  st.subheader('Visualising the demographics of each label')

  # Create a selectbox to choose a protected characteristic to explore
  selectbox_Lab = create_selectbox('Label to explore', ('Positive labels', 'Negative labels'))

  # Create a selectbox to choose a protected characteristic to explore
  selectbox_Char = create_selectbox('Characteristic to explore', characteristic_dict.keys())

  row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns((0.1, 1, 0.1, 1, 0.1, 1, 0.1))

  with row2_1:
    st.subheader("Model A")

    # Select test data
    if selectbox_Lab == 'Positive labels':
      data = df.loc[df['Model_A_label'] == 1]
    else:
      data = df.loc[df['Model_A_label'] == 0]
    
    # Use function plot_data to plot selected data
    plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])


  with row2_2:
    st.subheader("Model B")

    # Select test data
    if selectbox_Lab == 'Positive labels':
      data = df.loc[df['Model_B_label'] == 1]
    else:
      data = df.loc[df['Model_B_label'] == 0]

    # Use function plot_data to plot selected data
    plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])
  
  with row2_3:
    st.subheader("Model C")

    # Select test data
    if selectbox_Lab == 'Positive labels':
      data = df.loc[df['Model_B_label'] == 1]
    else:
      data = df.loc[df['Model_B_label'] == 0]

    # Use function plot_data to plot selected data
    plot_data(data, selectbox_Char, characteristic_dict[selectbox_Char])

### CREATE THE "MODEL PROPERTIES" PAGE ###

def mod_prop():
  st.subheader('Confusion matrices for all 3 models')
  tab1, tab2, tab3 = st.tabs(["Model A", "Model B", "Model C"])

  with tab1:
    st.header("Model A")
    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    with row2_1:
      create_confusion_matrix_heatmap(cm_A)
    with row2_2:
      plot_conf_rates(cm_A)

  with tab2:
    st.header("Model B")
    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    with row2_1:
      create_confusion_matrix_heatmap(cm_B)
    with row2_2:
      plot_conf_rates(cm_B)

  with tab3:
    st.header("Model C")
    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    with row2_1:
      create_confusion_matrix_heatmap(cm_C)
    with row2_2:
      plot_conf_rates(cm_C)

### CREATE THE "MODEL OUTCOMES" PAGE ###

def model_out():
  st.subheader('What do the different models select?')
  tab1, tab2, tab3 = st.tabs(["Characteristics", "Data", "Venn Diagram"])
      
  with tab1:
    row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space4 = st.columns((0.1, 1, 0.1, 1, 0.1, 1, 0.1))
    with row1_1:
      st.subheader("Model A")

      # Create a selectbox to choose a protected characteristic to explore
      selectbox = create_selectbox('Characteristic to explore - model A', characteristic_dict.keys())

      # Select test data
      data = df.loc[df['Predicted_A'] == "1"]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox, characteristic_dict[selectbox])

  with row1_2:
      st.subheader("Model B")

      # Create a selectbox to choose a protected characteristic to explore
      selectbox = create_selectbox('Characteristic to explore - model B', characteristic_dict.keys())

      # Select test data
      data = df.loc[df['Predicted_B'] == "1"]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox, characteristic_dict[selectbox])
  
  with row1_3:
      st.subheader("Model C")

      # Create a selectbox to choose a protected characteristic to explore
      selectbox = create_selectbox('Characteristic to explore - model C', characteristic_dict.keys())

      # Select test data
      data = df.loc[df['Predicted_C'] == "1"]

      # Use function plot_data to plot selected data
      plot_data(data, selectbox, characteristic_dict[selectbox])

  with tab2:
    selectbox_M = create_selectbox('Choose a model', pred_dict.keys())

    # Select data
    data = df.loc[df[pred_dict[selectbox_M]] == "1"]
    data = data.sort_values(by = prob_dict[selectbox_M], ascending = False)
    data = data[['Candidate ID','Prob_1_A', 'Prob_1_B', 'Prob_1_C', 'Predicted_A', 'Predicted_B', 'Predicted_C']]

    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(data.style.background_gradient(subset = ['Prob_1_A', 'Prob_1_B', 'Prob_1_C'], axis=0, vmin=0.80).highlight_max(color = '#FFCD9B', subset = ['Predicted_A', 'Predicted_B', 'Predicted_C'], axis=0))
    #text_gradient(cmap = 'winter', axis=0, subset = ['Prob_1_A', 'Prob_1_B', 'Prob_1_C'], vmin=0.70))
    #st.table(data.style.highlight_max(subset = ['Prob_1_A', 'Prob_1_B', 'Prob_1_C'], axis=0))

  with tab3:
    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    with row2_1:
      fig, ax = plt.subplots()

      list_A = df.loc[df['Predicted_A'] == "1", 'Candidate ID'].astype(int)
      list_B = df.loc[df['Predicted_B'] == "1", 'Candidate ID'].astype(int)
      list_C = df.loc[df['Predicted_C'] == "1", 'Candidate ID'].astype(int)
      set1 = set(list_A)
      set2 = set(list_B)
      set3 = set(list_C)

      venn3([set1, set2, set3], ('Model A', 'Model B', 'Model C'), ax=ax)
      st.pyplot(fig)

    with row2_2:
      st.markdown('comment on venn')


### ASSIGN WHAT TO DISPLAY ACCORDING TO THE SIDEBAR SELECTION ###
  
if add_radio == 'About':
  about()
if add_radio == "Data exploration":
  data_vis()
if add_radio == "Model confusion matrices":
  mod_prop()
if add_radio == "Model labels distribution":
  model_vis()
if add_radio == "Model outcomes":
  model_out()
if add_radio == "Model scores":
  data_plot()