#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:13:58 2023

@author: daliagala
"""

### LIBRARIES ###
import streamlit as st

### PAGE CONFIG ###
st.set_page_config(page_title='EquiVar', page_icon=':robot_face:', layout='wide')


#Set title
st.title('EquiVar - Target Variable Definition Simulator')

#Project description
st.subheader('Motivation')
st.markdown(
  """
  As machine techniques advance, algorithmic systems play an increasing role in hiring. For example, many such systems aim to predict which job applicants would be successful employees for a given job. In order for an algorithmic system to identify potentially successful employees, the notion of a successful employee must in some way be defined in terms that an algorithm can work with. In machine learning, this is called “defining the target variable” (in the case of hiring, the “target” that the algorithm is aiming at is a successful employee).
Defining the target variable is difficult. Imagine that you are hiring a salesperson. What makes for a good salesperson? Simply someone who makes the most profitable sales? Or is a good salesperson also a good leader? Does a good salesperson come up with new ideas that can improve how the sales team operates as a whole, and not just their individual sales? (The list could go on.) Perhaps the answer is: some of everything. But then we ask: how much of everything? How much more important are individual sales than leadership, for example? Put another way: there may be various different ways of understanding which qualities matter for being a good salesperson, and to what degree; there will likely be disagreement among hiring managers about what qualities matter (as anyone who’s been on a hiring committee has experienced). Even once it’s decided what makes for a good salesperson, there is a further question of how to make the notion precise in algorithmic terms: how do we identify job applicants with sales ability, leadership qualities, or innovative thinking? In order for the algorithm to be able to positively select those applicants, those qualities have to somehow be encoded in a numerical manner.
Defining the target variable is not only difficult; it can also have profound effects on fairness by resulting in disparate results for protected groups (like being a woman or being Black) and intersectional groups (like being a Black woman). For example, If you define the notion of a successful employee in one way you might end up hiring more women than if you were to define the notion in another way. Relatedly, machine learning models might behave differently depending on how successful employees are defined. Defining the notion in one way might lead to your model being less accurate for older applicants than for younger applicants. (There are many other ways that algorithmic decision-making can affect fairness and bias; the sorts of issues that arise from defining the target variable is just one of these (
  [Barocas and Selbst (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899)).

  """
  )
    
st.markdown(
  """The interactive dashboard we have developed is designed to help practitioners—those who build, procure, or deploy hiring algorithms—to understand, in concrete terms, how decisions about defining target variables impact fairness and bias. 
The dashboard has three parts:""")
st.markdown('''- **Define target variables.** You can define two different target variables for a “toy” hiring algorithm—two different ways of understanding what counts as a successful employee—by assigning different levels of cognitive characteristics (like reasoning skills, and information processing speed) that employees can have. Once you have defined the two target variables, our dashboard will, drawing on a data-set of the cognitive tests from real-world people, build two hiring models, A and B. Model A predicts which candidates will be successful employees on your first definition; model B predicts which candidates will be successful employees on your second definition.''')
st.markdown("- **See visualizations.** You can explore how the Datasets A and B generated based on your selections, and Models A and B trained on these datasets, differ in issues of fairness and bias. You can see, for example, which model selects more female applicants, or which model is more accurate for older applicants. You can also explore how Models A and B differ not just in matters of bias, but also in their overall performance: for example, in their overall accuracy.")
st.markdown("- **Putting the idea into practice.** A practitioner who is building or using their own hiring algorithms cannot take our dashboard “off the shelf” and apply it directly to their own data or algorithms. In this section, we explain how a practitioner could adapt our dashboard, and implement the ideas behind it, into their own work. [Note to Medb and Ray: this section is something we’re currently working on and is not in the current version of dashboard that you’re seeing].")



#Create info boxes for authors, links and GitHub
st.info('**Data Scientist: [Dalia Sara Gala](https://twitter.com/dalia_science)**')
st.info('**Philosophy Lead: [Milo Phillips-Brown](https://www.milopb.com/)**')
st.info('**Accenture team: [@The Dock](https://www.accenture.com/gb-en/services/about/innovation-hub-the-dock)**')
st.info('**GitHub: [Hiring-model](https://github.com/DaliaGala/Hiring-model)**')