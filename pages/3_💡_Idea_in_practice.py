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
st.markdown('''This dashboard is designed to help you understand how the notion of a “good employee” is translated into machine learning models—by defining target variables—and how target variable definition affects fairness in hiring algorithms (and other aspects of such algorithms too). On this page, we describe how to put the dashboard, and the insights it affords, into practice. 
How can this be done? The first step is to make two changes to the dashboard, because you cannot simply take it “off the shelf” and immediately put it into practice. This is because:''')

st.markdown('''-The dashboard is not built using your data or your models.''')

st.markdown('''-How you define a target variable in the dashboard is not how it’s done in practice. (Rather, the way you define it in the dashboard is a straightforward way for you to see the effects of target variable definition.)
Below we describe how to address these two issues.''')

st.subheader('''Using your data and your models''')

st.markdown('''The dashboard offers a starting point: if you want to build something like the dashboard for your data and your models, you now have a blue-print to work from.''')

st.subheader('''Defining the target variable in practice''')

st.markdown('''In the dashboard, you define the target variable by assigning weights to different cognitive characteristics. These weights determine the “positive label:” people in our dataset who perform best on the tests, given the weights you assign, are those assigned the positive label—that is, those that the model treats as “good.” Then, the model is trained to identify people whose cognitive characteristics match those with the positive label.''')

st.subheader("1. Get opinions of different hiring managers")


st.subheader('''2. Provide a broader starting group of "top employees"''')


st.markdown('''Such considerations are probably taken into account by the companies who provide hiring models to clients, but, to our knowlegde, insights of this sort are not published. Yet, our considerations further beg the question
of whether the majority of employees in the role for which a hiring model is being designed with the use of "top employees only" would be selected by such model. If our experiment showed that the model cannot distinguish between 
"top performers" and others, it would further lend legitimacy to the idea that target variable definition for hiring models is extremely subjective, and that such models can be designed with far more flexibility in the target variable
definition.''')
