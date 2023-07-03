#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:42:38 2023

@author: daliagala
"""

### IMPORT LIBRARIES ###
import streamlit as st

hide_st_style = """
            <style>
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')

### IDEA IN PRACTICE PAGE ###

st.title("Different target variable definitions in practice")
st.markdown('''This dashboard is designed to help you understand how the notion of a “good employee” is translated into machine learning models—by defining target variables—and how target variable definition affects fairness in hiring algorithms (and other aspects of such algorithms too). On this page, we describe how to put the dashboard, and the insights it affords, into practice.''')

st.markdown('''How can this be done? The first step is to make two changes to the dashboard, because you cannot simply take it “off the shelf” and immediately put it into practice. This is because:''')

st.markdown('''- The dashboard is not built using your data or your models.''')

st.markdown('''- How you define a target variable in the dashboard is not how it’s done in practice. (Rather, the way you define it in the dashboard is a straightforward way for you to see the effects of target variable definition.)''')
st.markdown('''Below we describe how to address these two issues.''')

st.subheader('''Using your data and your models''')

st.markdown('''The dashboard offers a starting point: if you want to build something like the dashboard for your data and your models, you now have a blue-print to work from.''')

st.subheader('''Defining the target variable in practice''')

st.markdown('''In the dashboard, you define the target variable by assigning weights to different cognitive characteristics. These weights determine the “positive label:” people in our dataset who perform best on the tests, given the weights you assign, are those assigned the positive label—that is, those that the model treats as “good.” Then, the model is trained to identify people whose cognitive characteristics match those with the positive label.''')

st.markdown('''As we discussed on the Home page, a growing number of hiring algorithms use cognitive tests to identify promising job applicants. However, the way target variable definition works with these real-world algorithms is different from how it works in the dashboard. For example, consider Pymetrics, a leading developer of hiring software. In some cases, Pymetrics builds bespoke algorithms for a company that is hiring for a given role. Pymetrics will ask the company to identify a group of the client’s current employees in that role that the client considers “good.” Then, these “good” employees play cognitive test games similar to the ones used in our dashboard. It is these employees who are assigned the positive labels. From this point on, Pymetrics’ algorithmic development goes just as it does in our dashboard: a model is trained to identify job applicants whose cognitive characteristics are similar to those with the positive label.''')

st.markdown('''So, for hiring algorithms like Pymetrics’, the target variable is defined not by assigning weights to cognitive attributes, but rather by directly identifying a certain group of current employees as “good.” In the dashboard, you can define different target variables by assigning different weights to the cognitive attributes. If you are in practice building an algorithm like Pymetrics’, you can define different target variables by identifying different groups of current employees as “good.”''')

st.markdown('''How might this work? As we discussed on the Home page of the dashboard, reasonable minds may disagree about what makes for a good employee, and relatedly reasonable minds may disagree about which current employees are good employees. For example, within a company, two different managers—call them Manager A and Manager B—may not be perfectly aligned in who they consider to be good employees for a certain role. The managers may agree in some cases. We might imagine that there are 50 employees whom both Manager A and Manager B deem good. But the two managers might disagree about other employees. Imagine that there are 25 further employees whom Manager A thinks of as good but Manager B does not (this needn’t mean that Manager B thinks that these employees are bad, just that they are not the best). Likewise, there might be 25 further employees whom Manager B thinks of as good but Manager A does not.''')

st.markdown('''In this case, there are two different (overlapping) groups of 75 employees, each corresponding to what Managers A and B think of as good employees. These two different groups of employees—and in turn, two different target variable definitions—could be used to train two different models.''')

st.markdown('''Instead of constructing two groups of “good” employees directly from the judgments of Managers A and B, you could weight their judgments against one another. For example, you could have two groups of employees, X and Y. Both X and Y contain the 50 employees that Managers A and B agree on. But group X contains 20 of Manager A’s preferred employees and 5 of Manager B’s, while group Y contains 20 of Manager B’s preferred employees and 5 of Manager A’s. Here again we have different groups of “good” employees, and so two different target variables.''')

st.markdown('''One could select different groups of good employees in other ways still. An employer might have different metrics to evaluate employee success. Different employees might be better than others according to one metric compared to another. Depending on what importance is assigned to the different metrics—depending on how you weight the different metrics against one another—different groups of employees may emerge as “good.”''')

st.markdown('''Our focus in the dashboard has been on hiring algorithms that are based on cognitive test games. There are other kinds of algorithms used in hiring—for example, algorithms that identify promising job applicants on the basis of their resumés. In designing any such algorithm, the target variable must be defined, and the notion of a “good employee” must be translated into algorithmic terms. And so the insights of this dashboard apply, and can be put into practice, for almost any kind of hiring algorithm you’re working with.''')