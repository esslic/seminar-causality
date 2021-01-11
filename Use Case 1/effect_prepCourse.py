# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:11:07 2020

@author: Christoph
"""
import dowhy
import pandas as pd
from sklearn import preprocessing

# Read data and rename columns
data = pd.read_csv("StudentsPerformance.csv", sep = ",")
data = data.rename(columns={'race/ethnicity': 'race', 'parental level of education': 'parental_level_of_education',
                     'test preparation course': 'test_preparation_course', 'math score': 'math_score',
                     'reading score': 'reading_score', 'writing score': 'writing_score'})

#%%
# Data preparation and transformation
data['gender'] = data['gender'].map({'female': 0, 'male': 1})
data['race'] = data['race'].map({'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3,'group E': 4})
data['parental_level_of_education'] = data['parental_level_of_education'].map({'some college': 0, 'some high school': 1,
                                                                               'high school': 2, 'associate\'s degree': 3,
                                                                               'bachelor\'s degree': 4, 'master\'s degree': 5})
data['lunch'] = data['lunch'].map({'standard': 0, 'free/reduced': 1})
data['test_preparation_course'] = data['test_preparation_course'].map({'none': 0, 'completed': 1})

data['test_preparation_course'] = data['test_preparation_course'].astype(bool)
data['overall_score'] = data['writing_score'] + data['reading_score'] + data ['math_score']
del data['writing_score']
del data['reading_score']
del data ['math_score']

#%%
# Defintion of causal graph
causal_graph = """digraph{
U[label="Unobserved_Confounders"];
U -> test_preparation_course; U -> overall_score; U -> parental_level_of_education; U -> lunch;
test_preparation_course -> overall_score;
gender -> test_preparation_course; gender -> overall_score; gender -> lunch; 
parental_level_of_education -> test_preparation_course; parental_level_of_education -> overall_score;
race -> parental_level_of_education;
}"""


model = dowhy.CausalModel(
    data, graph = causal_graph.replace("\n", " "), treatment = "test_preparation_course", outcome = "overall_score")

model.view_model()
from IPython.display import Image, display
display(Image(filename="causal_model.png"))

#%%
# Identify estimand
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

#%%
# Estimation of average treatment effect
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor1.propensity_score_matching",
                                target_units="ate")
print(estimate)


#%%
# Definition of different refutation methods
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                      placebo_type="permute", num_simulations=2)
print(refutation)



refutation = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print(refutation)

