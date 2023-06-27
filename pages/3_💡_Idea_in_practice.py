#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:42:38 2023

@author: daliagala
"""

### IMPORT LIBRARIES ###
import streamlit as st
import numpy as np
import pandas as pd 
import plotly.express as px

### IDEA IN PRACTICE PAGE ###

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
  st.image('./images/img1.jpg')

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

st.image('./images/img2.jpg')

st.markdown('''Such considerations are probably taken into account by the companies who provide hiring models to clients, but, to our knowlegde, insights of this sort are not published. Yet, our considerations further beg the question
of whether the majority of employees in the role for which a hiring model is being designed with the use of "top employees only" would be selected by such model. If our experiment showed that the model cannot distinguish between 
"top performers" and others, it would further lend legitimacy to the idea that target variable definition for hiring models is extremely subjective, and that such models can be designed with far more flexibility in the target variable
definition.''')
