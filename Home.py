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


#Create info boxes for authors, links and GitHub
st.info('**Data Scientist: [Dalia Sara Gala](https://twitter.com/dalia_science)**')
st.info('**Philosophy Lead: [Milo Phillips-Brown](https://www.milopb.com/)**')
st.info('**Accenture team: [@The Dock](https://www.accenture.com/gb-en/services/about/innovation-hub-the-dock)**')
st.info('**GitHub: [Hiring-model](https://github.com/DaliaGala/Hiring-model)**')