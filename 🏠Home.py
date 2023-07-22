#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:13:58 2023

@author: daliagala
"""

### LIBRARIES ###
import streamlit as st

### PAGE CONFIG ###
st.set_page_config(page_title='FairTargetSim', page_icon=':robot_face:', layout='wide')

hide_st_style = """
            <style>
            #GithubIcon {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#Set title
st.title('FairTargetSim')
st.subheader('''Interactive Target Variable Definition Simulator''')

#Project description
st.subheader('Motivation')

st.markdown('''As machine learning techniques advance, algorithmic systems play an increasing role in almost every domain of our lives. Machine learning requires translating real-world problems into mathematical representations. Sometimes, the translation is straightforward—for example, in predicting whether someone will default on a loan. Other times, things aren't so simple. Consider an algorithm that aims to predict which job applicants will be a good employee for a given job. In order for an algorithmic system to identify potentially good employees, the notion of a “good” employee must be defined in terms that an algorithm can work with. In machine learning, this is called “defining the target variable” (in the case of hiring, the algorithm “aims at the target” of finding good employees).''')

st.markdown('''Defining the target variable is difficult. Imagine that you are hiring a salesperson. What makes for a good salesperson? Simply someone who makes the most profitable sales? Or is a good salesperson also a good leader? Does a good salesperson come up with new ideas that can improve how the sales team operates as a whole, and not just their individual sales? (The list could go on.) Perhaps the answer is: some of everything.''')

st.markdown('''But then we ask: how much of each thing? How much more important are individual sales than leadership, for example? Put another way: there may be different ways of understanding which qualities matter for being a good salesperson, and to what degree; reasonable minds may disagree on these issues (as anyone who’s been on a hiring committee has experienced). Even once it’s decided what makes for a good salesperson, there is a further question of how to make the notion precise in algorithmic terms: how do we identify job applicants with sales ability, leadership qualities, or innovative thinking? In order for the algorithm to be able to positively select those applicants, those qualities have to somehow be encoded numerically.''')

st.markdown('''Defining the target variable is not only difficult; it can also have profound effects on fairness—by resulting in hiring disparities for protected groups [(Passi & Barocas 2019)](https://dl.acm.org/doi/10.1145/3287560.3287567). For example, if you define the notion of a “good” employee in one way you might end up hiring more women than if you were to define “good” in another way. Relatedly, machine learning models might behave differently depending on how “good” employee is defined. Defining the notion in one way might lead to your model being less accurate for older applicants than for younger applicants.''')

st.markdown('''These issues are not limited to hiring contexts. They arise in any case where there is no simple mathematical translation of a real-world problem. Take university admissions, for example. One might use an algorithm to predict which applicants will be “good” students.” What makes for a good student, though? A student who performs the best on exams at the end of their degree? A student who improves the most in their time at university? A student who doesn’t drop out, or who wins awards, or who gets a prestigious job after graduating, or contributes to the university in extracurricular activities? As with saying what makes for a good salesperson, the answer may be “some of everything,” and so again the question arises: how much of everything? Or consider another case: a news recommendation algorithm for an online platform. What makes for a “good” recommendation? Is it one that maximizes the user’s time on the platform, or that maximizes ad sales, or that is not “biased” along political lines (and then: which political lines?), or that best satisfies the user’s preferences, or that does not spread misinformation, or that prevents political polarization, or…? How these questions are answered—and how these different considerations are weighed against one another—has profound implications for fairness and other social and ethical concerns.''')

st.markdown('''Target variable definition, then, is not a merely technical matter. The question of what makes for a “good” employee, student, or news recommendation (and so on) is fundamentally value-laden. It calls for close attention and transparency [(Fazelpour & Danks, 2021)](https://compass.onlinelibrary.wiley.com/doi/full/10.1111/phc3.12760). All too often, though, target variables are defined in technical settings without attention to fairness. Further, stakeholders who aren't a part of the technical process—like managers in non-technical roles, or those working in upper management or human resources—either do not understand, or are simply not aware of, the fraught nature of target variable definition.''')

st.markdown('''We have developed FairTargetSim (FTS) to help address this issue. The simulator makes the implications of target variable definition explicit—and transparent—and offers a blue-print for those who want to address these effects in practice. FTS uses the case study of hiring; the lessons one can draw from it extend to any domain in which there are no clear-cut answers to the question of target variable definition.''')

st.subheader('Overview of the simulator')
st.markdown('''The simulator has three pages, which are best visited in order.''')
st.markdown('''- **Define Target Variables.** On this page, we invite you to imagine that you are building a hiring algorithm for a certain role. You can define two different target variables—two different ways of understanding what counts as a good employee. The simulator then uses your target variables to generate two datasets and two models. The first model predicts which candidates will be good employees according to your first definition of “good;” the second model predicts which candidates will be good employees according to your second definition.''')
st.markdown("- **Visualize The Results.** This page contains visualizations that illustrate how your two target variable definitions impact issues of fairness and overall model performance. You can see, for example, which model selects more female applicants, or which model is more accurate for older applicants. You can also see, among other things, how the two models differ in overall performance. In addition, you can see how your target variable definitions affect the data that go into training the model.")
st.markdown("- **Put the Idea into Practice.** This page contains guidance for putting the simulator, and the ideas behind it, into practice. A practitioner who is building or using their own hiring algorithms cannot take our simulator “off the shelf” and apply it directly to their own data or models. We give guidance for how a practitioner could adapt our simulator to use in their own work.")

st.subheader('Example')
st.markdown('''Below is an example of the simulator in action. On the Define Target Variables page, you’ll assign the importance of different cognitive characteristics by setting sliders that represent five different cognitive characteristics; you do this twice, and then the simulator builds two models, A and B. On the Visualize the Results page, you’ll see how even very small changes—such as changing one point of importance for “behavioral restraint’’ (highlighted in green)—can result in completely different outcomes for the models.''')
st.markdown('''**From the Define Target Variables page:**''')
st.image('./images/tests.png')

st.markdown('''**From the Visualize the Results page:**''')
st.image('./images/pie_charts.png')