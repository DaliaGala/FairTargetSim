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
from sklearn.metrics import accuracy_score
from statkit.non_parametric import bootstrap_score
from utils import assign_labels_by_probabilities, drop_data_exp_2, train_and_predict, display_proportional, plot_data, run_PCA, create_confusion_matrix_heatmap, plot_conf_rates, mod_prop

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
          'education level' : colours_education,
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

protected_chars = ['education level', 'country', 'age', 'gender']
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
dataframe = dataframe.rename(columns={"education_level": "education level"})

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')

### CREATE THE "ABOUT" PAGE ###
def about():
  #Set title
  st.title('EquiVar - Target Variable Definition Simulator')

  #Project description
  st.subheader('Motivation')
  st.write(
      """
            Hiring applications often require for recruiters and employers to sift through a large volume of applications. With the progress in the development of more sophisticated machine learning algorithms, companies are becoming more reliant on such systems to produce lists of candidates who could be potential hires. These algorithms can, for example, rely on keyword matching for candidate CVs and job requirements. Recently, a new type of algorithms are becoming more extensively used, which try to deploy games, riddles and challenges to capture some of the candidates' qualities. Through these, companies can assess the candidates' capacities by comparing their results to those of successful incumbent successful employees.

A concern which emerges around the concept of a "successful employee" in the hiring context is the following: how does one define what a successful employee is? There might be **multiple ways** of defining this. As [Barocas and Selbst (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899) outline: _“Good” must be defined in ways that correspond to measurable outcomes: relatively higher sales, shorter production time, or longer tenure, for example._ The model creators must then _translate some amorphous problem into a question that can be expressed in more formal terms that computers can parse_, which is often an ambiguous task. Therefore, the definition of a "good employee" or the way in which the **target variable** is specified, hugely impacts the model outcomes.

As such, any social application of such technologies carries with it risks related to fairness and bias. Inevitably, machine learning models are but mere generalisations and, in issues as complex as hiring, their use can lead to disparate outcomes for protected groups and minorities. Moreover, some of these algorithms can inadvertently become trained to select candidates based on a feature directly correlated with some of their protected characteristics, called "selection by proxy". Such selection is illegal in many countries in the world, where equal employment opportunity laws prevent the employers from making their decisions based on sex, race, religion or belief, ethnic or national origin, disability, age, citizenship, marital, domestic or civil partnership status, sexual orientation, gender identity, pregnancy or related condition (including breastfeeding) or any other basis as protected by applicable law. 

Usually, to address these issues, constraints are put on the models which are being trained and deployed. These could take the form of, for example, mandating that the proportions of candidates selected by the model be correlated with the proportions of the input populations, etc. However, such measures often result in **accuracy versus fairness** problems. This describes a situation in which achieving better fairness by some specified fairness metric results in lower accuracy of the model.

To tackle these problems, we have designed a new paradigm for hiring by the machine, which we called Equi Var, for Equivalent Target Variable. This work was carried out to achieve the following goals:

1. Explain that the target variable is usually defined in an implicit way, and create a simulator to do so explicitly and in different ways.

2. Demonstrate that hiring algorithms trained on these target variables, though nominally equivalent, can produce differing outcomes with respect to protected characteristics of assessed candidates.

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
st.sidebar.subheader("Contents")

# Add selection to the sidebar
with st.sidebar:
    add_radio = st.radio(
        "",
        ("About", "Define the target variables", "Visualise your models", "The idea in practice")
    )
    st.sidebar.subheader("Authors")
    #Create info boxes for authors, links and GitHub
    st.info('**Data Scientist: [Dalia Sara Gala](https://twitter.com/dalia_science)**')
    st.info('**Philosophy Lead: [Milo Phillips-Brown](https://www.milopb.com/)**')
    st.info('**Accenture team: [@The Dock](https://www.accenture.com/gb-en/services/about/innovation-hub-the-dock)**')
    st.info('**GitHub: [Hiring-model](https://github.com/DaliaGala/Hiring-model)**')

### CREATE THE "DATA VISUALISATION" PAGE ###

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

def data_vis(full_df, dataframe_PCA):
  st.markdown('''On this page, you can see the distribution of the dataset labels which were assigned based on the scores calculated from the slider values you selected previously. 
  [Principal Components Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), or PCA, is a technique often used to analyse and subsequently visualise datasets 
  where there are many features per single example. This is the case with the NCPT dataset used in our simulator. Specifically, the battery which we used has 11 features per single 
  example, the example being the player of cognitive games, and, in our metaphor, a hypothetical employee or job candidate. It is impossible to plot 11 dimensions, and PCA allows 
  for the visualisation of multidimensional data, while also preserving as much information as possible.''')
  add_radio = st.radio("Choose what to explore", ("Principal Components Analysis", "Component loadings"), horizontal=True)

  if add_radio == "Principal Components Analysis":
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
    with row1_1:
      st.write("Model A PCA")
      pcaA, dfA, labelsA, coeffA, componentsA = run_PCA(dataframe_PCA, 'Model_B_label', 'Model_A_label', 3)
      total_var = pcaA.explained_variance_ratio_.sum() * 100

      fig = px.scatter_3d(
          componentsA, x=0, y=1, z=2, color=dfA['target'],
          labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, 
          title=f'Total Explained Variance: {total_var:.2f}%'
      )
      fig.update_traces(marker_size = 5)
      st.plotly_chart(fig)
    with row1_2:
      st.write("Model B PCA")
      pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 3)

      fig = px.scatter_3d(
          componentsB, x=0, y=1, z=2, color=dfB['target'],
          labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
      )
      fig.update_traces(marker_size = 5)
      st.plotly_chart(fig)
    st.markdown('''These plots show the reduction of 11 dimensions (11 subtest results) to 3 dimensions. At the top, you can see how much information, or "Total Variance", has
    been preserved. Note that for both datasets, A and B, different points are labelled "1" or "0". This shows that the two datasets represent the two different target variable
    definitions which were created by you previously. The plots are interactive - zoom in to explore in detail.''')

  if add_radio == "Component loadings":
    pcaB, dfB, labelsB, coeffB, componentsB = run_PCA(dataframe_PCA, 'Model_A_label', 'Model_B_label', 2)
    loadings = pcaB.components_.T * np.sqrt(pcaB.explained_variance_)
    total_var = pcaB.explained_variance_ratio_.sum() * 100

    fig = px.scatter(componentsB, x=0, y=1, title=f'Total Explained Variance: {total_var:.2f}%')
    fig.update_layout(yaxis_title="PC2", xaxis_title="PC1")

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
    
    st.markdown('''On this plot, PCA component loadings can be explored. These facilitate the understanding of how much each variable (which there are 11 of)
    contributes to a particular principal component. Here, the 11 variables were reduced to 2 components, which are labelled PC1 and PC2 on the x and y axes.
    The magnitude of the loading (here displayed as an arrow per each variable) indicates how strong the relationship between the variable and the component is.
    Therefore, the longer the arrow, the stronger the relationship between that component and that variable. The loading's sign can be positive or negative. 
    This indicates whether the principal component and that variable are positively or negatively correlated. We can see that multiple variables are positively
    correlated with PC2. Two variables, episodic verbal learning and delayed recall are negatively correlated with both of the components, which means that the 
    variance in them is not well represented by these two components.''')

def model_out(full_df):
  st.markdown('''This section shows you the differences between your two models when they are given the same set of previously unseen candidates to assign labels to. You are exploring those candidates who were given label "1" by the models,
  therefore, they would be the ones offered the job interview or selected by the position based on your target variable definition.''')
  add_radio = st.radio("Choose what to explore", ("Demographics", "See the output data", "Venn diagram"), horizontal=True)
  if add_radio == "Demographics":
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

  if add_radio == "See the output data":
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

  if add_radio == "Venn diagram":
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
    page_radio = st.radio('What would you like to see?', ("Modelling outcomes", "Input scores", "Label distribution", "Principal Component Analysis", "Outcomes per group"), horizontal=True)
    dataframe = st.session_state[key1]
    clean_data = st.session_state[key2]
    if page_radio == "Input scores":
      model_scores(dataframe)
    if page_radio == "Principal Component Analysis":
      data_vis(dataframe, clean_data)
    if page_radio == "Modelling outcomes":
      model_out(dataframe)
    if page_radio == "Label distribution":
      model_vis(dataframe)
    if page_radio == "Outcomes per group":
      filter_for_protected(dataframe)

### CREATE THE "TARGET VARIABLE DEFINITION" PAGE ###
def define_target_variable():

  st.title('Target variable definition')
  # expander = st.expander("See explanation")
  
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
    of being selected the higher their score was. This is done to emulate what a real-world decision might look like, and to introduce noise to the model and prevent it from being 100% accurate. The models will then be
    trained, and you will see output metrics for each model. They will include: how many candidates labelled "1" are shared between the two datasets based on your selections, model accuracy and confidence interval of the accuracy,
    as well as model [confusion matrices](https://en.wikipedia.org/wiki/Confusion_matrix) and tables containing model [metrics](https://en.wikipedia.org/wiki/Precision_and_recall). Once you analysed these, 
    head to "Visualise your models" in the sidebar to start exploring the demographic differences between your models.''')

  col1, col2 = st.columns(2)

  selectionsA = {}
  selectionsB = {}

  groups = ["attention", "reasoning", "memory", "behavioural restraint", "information processing speed"]
  results_dict_A = groups_dict
  results_dict_B = groups_dict

  with col1:
    st.subheader("Define target variable for model A ")
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
    st.subheader("Define target variable for model B ")
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


      clean_data = drop_data_exp_2(new_annotated)
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

      mod_prop(cm_A, cm_B)

      st.success('''Success! You have defined the target variables and trained your models. Click "Visualise your models" in the sidebar to explore.''')

def idea_practice():
  st.title("Different target variable definitions in practice")
  st.markdown('''What we presented here is a simulator which allows stakeholders and practitioners to understand the importance which target variable definition has on
  the modelling outcomes. Because this process is usually implicit, as we discussed, there is a general lack of clarity about how subjective, amorphous judgements are
  translated into numerical outcomes which define the success or failure of job applicants. This process is often so complicated that it might require a high level of understanding
  of data science and machine learning practices to comprehend fully.''')

  st.markdown('''One of our goals was to make explicit something which currently is implicit in business practice. The judgements of hiring managers who nominate "top employees"
  based on which the ideal candidate profile will be modelled do so "in their heads", and the choices which they make are effectively absolute. Once the manager decided who will be "1"
  and who will be "0", the model is trained, and this choice is solidified in a numerical manner. We make this process explicit and allow the user of this website to define two different
  target variables with the use of sliders, where their choices will be translated into labels for two different models. In an interactive manner the user can then see how their choices
  can make the outcomes of the modelling vastly different for the two models. Thus, we hope to present the paradigm of radical explainability of machine learning models. In our simulator,
  one can see easily and plainly how choices with respect to target variable definition affect the model selections, and try to adjust these choices to achieve a more fair outcome.''')

  st.markdown('''Our simulator, naturally, is just that - a visualisation tool. How then can the concepts of target variable definition be used in the real world? As we explored here, 
  the way that the target variable is defined is concretised in the model labels. Each model is defined by its labels, or the "ground truth", based on which it learns how to label the examples
  which it is presented with. Thus, the only way to implement this concept of varying target variables in practice is by having multiple "ground truths", or multiple models. It could be 
  counterintuitive for some models unrelated to hiring. If we are trying to distinguish butterflies from dragonflies, there is not much debate about which is which. The examples are labelled
  and the model uses the features to classify the insects. In the case of any subjective target variable, particularly those related to practices surrounding credithworthiness, hiring, college
  admissions etc., the classification is less than obvious, and the definition of "good" is subjective. After all, many practitioners and managers will say that they faced great difficulty in 
  selecting the candidates for interviews, or ones to hire after interviews. If a panel is making this decision, often they will agree on some of the candidates who they want to hire, but cannot
  reach consensus about the remaining ones, with every panel member placing empahsis on different qualities of the interviewed candidates. We therefore propose two ways in which the concept of target 
  variable definition can be used to improve the modelling outcomes and allow for choice leading to fairer employment outcomes.''')

  st.subheader("1. Get opinions of different hiring managers")

  c1, c2 = st.columns((2, 2))

  with c1:
    st.image('images/img1.jpg')

  with c2:
    st.markdown('''The first one would be to indeed ask multiple hiring managers for their opinions on which employees to model the hiring algorithm after. It is more than likely that each of them will name some 
  of the same employees, but equally, the groups which they compose might differ significantly. Mutliple models can then be trained, and used on a case by case basis to achieve required parity.
  Though model training can be expensive, practices are already in place which ensure that machine learning models produce fair outcomes and if they do not, they have to be re-trained. By training
  more models from the get-go, the iterative process of training and retraining models can be avoided, and the fairest model of the ones which were trained can be used. Moreover, the process by which
  model fairness is achieved often involves the accuracy versus fairness tradeoff. By having mutliple target variables, and therefore multiple models to start with, it might be possible to achieve models
  of similar accuracy while one of them will be more fair than the others. Therefore, the tradeoff will not be necessary anymore, and the most fair model can be chosen.''')
  
  st.subheader('''2. Provide a broader starting group of "top employees"''')

  c1, c2 = st.columns((3, 2))
  with c1:
    st.markdown('''It is however also understandable that asking multiple managers to do the same job could be considered excessive. Many companies want to save money and remain focused, and this approach
  might not be convincing to them. Therefore, we suggest a second solution. The way in which the hiring models are currently being build often includes some top down instructions on what kinds of equal 
  representation requirements the models must meet. This often results in an iterative process, where if a model does not meet these requirements, more "top employees" are nominated by the client to create
  a better model. Therefore, the initial group of "top employees" could become widened. We suggest to include a much larger group of employees in the initial group. This would allow the data scientists building
  the model to iteratively adjust the target variable, and build the fairest possible model on a subset of the larger group of candidates. It would save the client time by removing the need to add to the group of
  "top employees" if the model does not meet the fairness requirements, and would allow some flexibility in the target variable definition. This has been demonstrated on the graph below. "All candidates" indicates
  a wide group of "top employees" provided by the client. "Multiple target variables" demonstrates how the data scientists who are building the models could divide this wider group into three subgroups, A, B and C,
  build and test three different models (or more, as now there is choice in terms of target variable definition), and deliver the most fair model to the customer.''')

  with c2:
    x = np.random.normal(5.0, 1.0, 1000) 
    y = np.random.normal(10.0, 2.0, 1000)

    df = pd.DataFrame({'PC1':x, 'PC2':y})
    conditions = [
      (df['PC1'] >= 5) & (df['PC2'] < 9),
      (df['PC1'] >= 5) & (df['PC2'].between(9, 11)),
      (df['PC1'] > 5) & (df['PC2'] >= 11)
      ]
    values = ['A', 'B', 'C']
    df['Group'] = 0
    df['Group'] = np.select(conditions, values)
    df['General values'] = "0"
    df.loc[df['Group'] != "0", 'General values'] = "1"

    orderedDF = df.sort_values(by="Group", ascending=True)
    if "unchanged_df" not in st.session_state:
      st.session_state.unchanged_df = orderedDF
    
    view = st.radio("Choose view", ("All candidates", "Multiple target variables"))
    if view == "All candidates":
      fig = px.scatter(x = st.session_state.unchanged_df["PC1"], y = st.session_state.unchanged_df["PC2"], color = st.session_state.unchanged_df["General values"])
    else:
      fig = px.scatter(x = st.session_state.unchanged_df["PC1"], y = st.session_state.unchanged_df["PC2"], color = st.session_state.unchanged_df["Group"])
    st.plotly_chart(fig, use_container_width=True)

  st.subheader('''Or...3. Are "top employees" indeed different from other employees?''')

  st.markdown('''Our considerations, however, bring us to an important question. Is there truly a **numerically defineable difference** between the top employees in a given role, and the rest of the employees of that company in that role?
  Data of this sort was not available to us, but we propose the following experiment:''')
  
  st.image('images/img2.jpg')

  st.markdown('''Such considerations are probably taken into account by the companies who provide hiring models to clients, but, to our knowlegde, insights of this sort are not published. Yet, our considerations further beg the question
  of whether the majority of employees in the role for which a hiring model is being designed with the use of "top employees only" would be selected by such model. If our experiment showed that the model cannot distinguish between 
  "top performers" and others, it would further lend legitimacy to the idea that target variable definition for hiring models is extremely subjective, and that such models can be designed with far more flexibility in the target variable
  definition.''')


### ASSIGN WHAT TO DISPLAY ACCORDING TO THE SIDEBAR SELECTION ###
  
if add_radio == 'About':
  about()
if add_radio == "Define the target variables":
  define_target_variable()
if add_radio == "Visualise your models":
  data_plot('complete_df', 'clean_df', 'cm_A', 'cm_B')
if add_radio == "The idea in practice":
  idea_practice()