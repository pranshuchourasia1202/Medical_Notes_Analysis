from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
import locale
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, fbeta_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel, BioGptModel, AutoModelForSequenceClassification, AdamW, AutoModelForCausalLM, BioGptForCausalLM,GPT2ForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, TensorDataset
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")
import gc
torch.cuda.empty_cache()
gc.collect()
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import sys

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth",500)
pd.set_option("display.max_info_columns",50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# The following code extracts a specific file from a zip archive and saves it in a specific location.
with ZipFile('./MedicalNotesNLPChallenge.zip', 'r') as zObject:
    # Extracting specific file in the zip
    # into a specific location.
    zObject.extractall(
        path="./MedicalNotesNLPChallenge/")
zObject.close()

# Reading two csv files containing medical terms and notes respectively.
term_matching = pd.read_csv('./MedicalNotesNLPChallenge/MedicalConcepts.csv')
medical_notes = pd.read_csv('./MedicalNotesNLPChallenge/ClinNotes.csv')

# Printing the shape of the two dataframes.
print(term_matching.shape)
print(medical_notes.shape)

# Removing duplicate rows from the term_matching dataframe.
term_matching = term_matching.drop_duplicates().reset_index(drop=True)
print(term_matching.shape)

# Obtaining components from the term_matching dataframe.
components = get_components(term_matching)
print(len(components))
print(components)

# Creating a list of tuples containing word pairs from the term_matching dataframe.
word_pairs = [(first,second) for first,second in zip(term_matching['Term1'],term_matching['Term2'])]

# Defining a dictionary containing model names, their corresponding models and tokenizers.
model_dict={'BioBERT':{'Model':'dmis-lab/biobert-base-cased-v1.1','Tokenizer':'dmis-lab/biobert-base-cased-v1.1'},
            'ClinicalBERT':{'Model':'emilyalsentzer/Bio_ClinicalBERT','Tokenizer':'emilyalsentzer/Bio_ClinicalBERT'}, 
            'BlueBERT':{'Model':'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12','Tokenizer':'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'}, 
            'BioGPT':{'Model':'microsoft/biogpt','Tokenizer':'microsoft/biogpt'}
            }

# Defining a dictionary containing medical categories and their corresponding numerical labels.
label_dict = {'Cardiovascular / Pulmonary':0,'Gastroenterology':1,'Neurology':2}

# Preprocessing the medical notes dataframe and mapping the medical categories to numerical labels.
medical_notes,label_dict = preprocess_notes(medical_notes,label_dict)

#  Analyzing the models using the word_pairs and medical_notes dataframes.
medical_notes, term_matching, AllResult = Analyze_models(model_dict,word_pairs,medical_notes,term_matching,label_dict)


#  Saving the term_matching and medical_notes dataframes to csv files.
term_matching.to_csv('term_matching.csv', index=False)
medical_notes.to_csv('medical_notes.csv',index=False)

#  Saving the AllResult object to a pickle file.
with open('AllResult.pickle', 'wb') as handle:
    pickle.dump(AllResult, handle, protocol=pickle.HIGHEST_PROTOCOL)


# For generating lime prediction
# Defining a smaller name dictionary (for Lime) containing medical categories and their corresponding numerical labels.
label_dict_small = {'Cardio/Pul': 0, 'Gastro': 1, 'Neuro': 2}

# Set of text and its original labels (to be given for explainability)
texts = ["clinical indication chest pain interpretation patient receive 14.9 mci cardiolite rest portion study 11.5 mci cardiolite stress portion study the patient baseline ekg normal sinus rhythm patient stress accord bruce protocol dr. x. exercise test supervise interpret dr. x. separate report stress portion study the myocardial perfusion spect study show mild anteroseptal fix defect see likely secondary soft tissue attenuation artifact mild partially reversible perfusion defect see pronounced stress image short axis view suggestive minimal ischemia inferolateral wall the gate spect study show normal wall motion wall thickening calculate leave ventricular ejection fraction 59%.conclusion:1 exercise myocardial perfusion study show possibility mild ischemia inferolateral wall 2 normal lv systolic function lv ejection fraction 59",
         "preoperative diagnosis abdominal mass postoperative diagnosis abdominal mass procedure paracentesis description procedure 64 year old female stage ii endometrial carcinoma resect treat chemotherapy radiation present time patient radiation treatment week ago develop large abdominal mass cystic nature radiologist insert pigtail catheter emergency room proceed admit patient drain significant clear fluid subsequent day cytology fluid negative culture negative eventually patient send home pigtail shut patient week later undergo repeat cat scan abdoman pelvis the cat scan show accumulation fluid mass achieve 80 previous size call patient home come emergency department service provide time proceed work pigtail catheter obtain informed consent prepare drape area usual fashion unfortunately catheter open drainage system time withdraw directly syringe 700 ml clear fluid system connect drain bag patient instruct log use equipment give appointment office monday day",
         "technique sequential axial ct image obtain cervical spine contrast additional high resolution coronal sagittal reconstruct image obtain well visualization osseous structure findings cervical spine demonstrate normal alignment mineralization evidence fracture dislocation spondylolisthesis vertebral body height disc space maintain central canal patent pedicle posterior element intact paravertebral soft tissue normal limit atlanto den interval den intact visualized lung apex clear impression acute abnormality"
        ]
correct_labels = [0,1,2]

# Saved model names
saved_models_raw = ['BioBERT_notes','ClinicalBERT_notes','BlueBERT_notes','BioGPT_notes']
saved_models_preprocess = ['BioBERT_notes_preprocess','ClinicalBERT_notes_preprocess','BlueBERT_notes_preprocess','BioGPT_notes_preprocess']

# Used model for lime
model_name = 'BioBERT_notes_preprocess'
num = 1
# Path where lime results are saved as .html file
save_path = './lime/'+model_name+"_"+str(num)+".html"
if not os.path.exists('./lime/'):
        os.makedirs('./lime/')

# Calling lime prediction function
lime_prediction(model_name, texts[num], save_path,label_dict_small, output_label = (correct_labels[num],), num_features=6, num_samples = 50)




