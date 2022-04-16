
from django.shortcuts import render
import requests
import keras
#importing the necessary libraries
import numpy as np 
import pandas as pd 

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import copy 
import os
from datetime import datetime

path=os.getcwd()+'\pred\static\pred\disease_model.h5'
print(path)
DISEASE_MODEL = keras.models.load_model(path)

# create test for prediction
SYMPTOMS = {'itching': {0: 0}, 'skin_rash': {0: 0}, 'nodal_skin_eruptions': {0: 0}, 'continuous_sneezing': {0: 0}, 'shivering': {0: 0}, 'chills': {0: 0}, 'joint_pain': {0: 0}, 'stomach_pain': {0: 0}, 'acidity': {0: 0}, 'ulcers_on_tongue': {0: 0}, 'muscle_wasting': {0: 0}, 'vomiting': {0: 0}, 'burning_micturition': {0: 0}, 'spotting_ urination': {0: 0}, 'fatigue': {0: 0}, 'weight_gain': {0: 0}, 'anxiety': {0: 0}, 'cold_hands_and_feets': {0: 0}, 'mood_swings': {0: 0}, 'weight_loss': {0: 0}, 'restlessness': {0: 0}, 'lethargy': {0: 0}, 'patches_in_throat': {0: 0}, 'irregular_sugar_level': {0: 0}, 'cough': {0: 0}, 'high_fever': {0: 0}, 'sunken_eyes': {0: 0}, 'breathlessness': {0: 0}, 'sweating': {0: 0}, 'dehydration': {0: 0}, 'indigestion': {0: 0}, 'headache': {0: 0}, 'yellowish_skin': {0: 0}, 'dark_urine': {0: 0}, 'nausea': {0: 0}, 'loss_of_appetite': {0: 0}, 'pain_behind_the_eyes': {0: 0}, 'back_pain': {0: 0}, 'constipation': {0: 0}, 'abdominal_pain': {0: 0}, 'diarrhoea': {0: 0}, 'mild_fever': {0: 0}, 'yellow_urine': {0: 0}, 'yellowing_of_eyes': {0: 0}, 'acute_liver_failure': {0: 0}, 'fluid_overload': {0: 0}, 'swelling_of_stomach': {0: 0}, 'swelled_lymph_nodes': {0: 0}, 'malaise': {0: 0}, 'blurred_and_distorted_vision': {0: 0}, 'phlegm': {0: 0}, 'throat_irritation': {0: 0}, 'redness_of_eyes': {0: 0}, 'sinus_pressure': {0: 0}, 'runny_nose': {0: 0}, 'congestion': {0: 0}, 'chest_pain': {0: 0}, 'weakness_in_limbs': {0: 0}, 'fast_heart_rate': {0: 0}, 'pain_during_bowel_movements': {0: 0}, 'pain_in_anal_region': {0: 0}, 'bloody_stool': {0: 0}, 'irritation_in_anus': {0: 0}, 'neck_pain': {0: 0}, 'dizziness': {0: 0}, 'cramps': {0: 0}, 'bruising': {0: 0}, 'obesity': {0: 0}, 'swollen_legs': {0: 0}, 'swollen_blood_vessels': {0: 0}, 'puffy_face_and_eyes': {0: 0}, 'enlarged_thyroid': {0: 0}, 'brittle_nails': {0: 0}, 'swollen_extremeties': {0: 0}, 'excessive_hunger': {0: 0}, 'extra_marital_contacts': {0: 0}, 'drying_and_tingling_lips': {0: 0}, 'slurred_speech': {0: 0}, 'knee_pain': {0: 0}, 'hip_joint_pain': {0: 0}, 'muscle_weakness': {0: 0}, 'stiff_neck': {0: 0}, 'swelling_joints': {0: 0}, 'movement_stiffness': {0: 0}, 'spinning_movements': {0: 0}, 'loss_of_balance': {0: 0}, 'unsteadiness': {0: 0}, 'weakness_of_one_body_side': {0: 0}, 'loss_of_smell': {0: 0}, 'bladder_discomfort': {0: 0}, 'foul_smell_of urine': {0: 0}, 'continuous_feel_of_urine': {0: 0}, 'passage_of_gases': {0: 0}, 'internal_itching': {0: 0}, 'toxic_look_(typhos)': {0: 0}, 'depression': {0: 0}, 'irritability': {0: 0}, 'muscle_pain': {0: 0}, 'altered_sensorium': {0: 0}, 'red_spots_over_body': {0: 0}, 'belly_pain': {0: 0}, 'abnormal_menstruation': {0: 0}, 'dischromic _patches': {0: 0}, 'watering_from_eyes': {0: 0}, 'increased_appetite': {0: 0}, 'polyuria': {0: 0}, 'family_history': {0: 0}, 'mucoid_sputum': {0: 0}, 'rusty_sputum': {0: 0}, 'lack_of_concentration': {0: 0}, 'visual_disturbances': {0: 0}, 'receiving_blood_transfusion': {0: 0}, 'receiving_unsterile_injections': {0: 0}, 'coma': {0: 0}, 'stomach_bleeding': {0: 0}, 'distention_of_abdomen': {0: 0}, 'history_of_alcohol_consumption': {0: 0}, 'fluid_overload.1': {0: 0}, 'blood_in_sputum': {0: 0}, 'prominent_veins_on_calf': {0: 0}, 'palpitations': {0: 0}, 'painful_walking': {0: 0}, 'pus_filled_pimples': {0: 0}, 'blackheads': {0: 0}, 'scurring': {0: 0}, 'skin_peeling': {0: 0}, 'silver_like_dusting': {0: 0}, 'small_dents_in_nails': {0: 0}, 'inflammatory_nails': {0: 0}, 'blister': {0: 0}, 'red_sore_around_nose': {0: 0}, 'yellow_crust_ooze': {0: 0}}

LABELS = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']

def get_symp_pd(symptoms_list):
    s_dict = copy.deepcopy(SYMPTOMS)
    for symp in symptoms_list:
        if symp in s_dict:
            s_dict.get(symp)[0] = 1
    # convert symptoms to pandas data frame
    s_dict = pd.DataFrame(s_dict)
    
    return s_dict

def get_disease_name(symptoms_list):
    symptoms_list = get_symp_pd(symptoms_list)
    y_prob = DISEASE_MODEL.predict(symptoms_list) 

    sorted_index_array=np.argsort(y_prob[0])

    sorted_array=y_prob[0][sorted_index_array]
    n=3
    rslt=sorted_array[-n: ]

    y_classes = y_prob.argmax(axis=-1)

    return LABELS[y_prob[0].argmax(axis=-1)],rslt,y_prob

# symptom_map={
#     'Itching':'itching',
#      'Skin Rash':'skin_rash',
#      'Nodal Skin Eruptions':'nodal_skin_eruptions',
#      'Continuous Sneezing':'continuous_sneezing',
#      'Shivering':'shivering',
#      'Chills':'chills',
#      'Joint Pain':'joint_pain',
#      'Stomach Pain':'stomach_pain',
#      'Acidity':'acidity',
#      'Ulcers On Tongue':'ulcers_on_tongue',
#      'Muscle Wasting':'muscle_wasting',
#      'Vomiting':'vomiting',
#      'Burning Micturition':'burning_micturition',
#      'Spotting  Urination':'spotting_urination',
#      'Fatigue':'fatigue',
#      'Weight Gain':'weight_gain',
#      'Anxiety':'anxiety',
#      'Cold Hands And Feets':'cold_hands_and_feets',
#      'Mood Swings':'mood_swings',
#      'Weight Loss':'weight_loss',
#      'Restlessness':'restlessness',
#      'Lethargy':'lethargy',
#      'Patches In Throat':'patches_in_throat',
#      'Irregular Sugar Level':'irregular_sugar_level',
#      'Cough':'cough',
#      'High Fever':'high_fever',
#      'Sunken Eyes':'sunken_eyes',
#      'Breathlessness':'breathlessness',
#      'Sweating':'sweating',
#      'Dehydration':'dehydration',
#      'Indigestion':'indigestion',
#      'Headache':'headache',
#      'Yellowish Skin':'yellowish_skin',
#      'Dark Urine':'dark_urine',
#      'Nausea':'nausea',
#      'Loss Of Appetite':'loss_of_appetite',
#      'Pain Behind The Eyes':'pain_behind_the_eyes',
#      'Back Pain':'back_pain',
#      'Constipation':'constipation',
#      'Abdominal Pain':'abdominal_pain',
#      'Diarrhoea':'diarrhoea',
#      'Mild Fever':'mild_fever',
#      'Yellow Urine':'yellow_urine',
#      'Yellowing Of Eyes':'yellowing_of_eyes',
#      'Acute Liver Failure':'acute_liver_failure',
#      'Fluid Overload':'fluid_overload',
#      'Swelling Of Stomach':'swelling_of_stomach',
#      'Swelled Lymph Nodes':'swelled_lymph_nodes',
#      'Malaise':'malaise',
#      'Blurred And Distorted Vision':'blurred_and_distorted_vision',
#      'Phlegm':'phlegm',
#      'Throat Irritation':'throat_irritation',
#      'Redness Of Eyes':'redness_of_eyes',
#      'Sinus Pressure':'sinus_pressure',
#      'Runny Nose':'runny_nose',
#      'Congestion':'congestion',
#      'Chest Pain':'chest_pain',
#      'Weakness In Limbs':'weakness_in_limbs',
#      'Fast Heart Rate':'fast_heart_rate',
#      'Pain During Bowel Movements':'pain_during_bowel_movements',
#      'Pain In Anal Region':'pain_in_anal_region',
#      'Bloody Stool':'bloody_stool',
#      'Irritation In Anus':'irritation_in_anus',
#      'Neck Pain':'neck_pain',
#      'Dizziness':'dizziness',
#      'Cramps':'cramps',
#      'Bruising':'bruising',
#      'Obesity':'obesity',
#      'Swollen Legs':'swollen_legs',
#      'Swollen Blood Vessels':'swollen_blood_vessels',
#      'Puffy Face And Eyes':'puppy_face_and_eyes',
#      'Enlarged Thyroid':'enlarged_thyroid',
#      'Brittle Nails':'brittle_nails',
#      'Swollen Extremeties':'swollen_extremeties',
#      'Excessive Hunger':'excessive_hunger',
#     'Extra Marital Contacts':'extra_marital_contacts',
#     'Drying And Tingling Lips':'drying_and_tingling_lips',
#     'Slurred Speech':'slurred_speech',
#     'Knee Pain':'knee_pain',
#     'Hip Joint Pain':'hip_joint_pain',
#     'Muscle Weakness':'muscle_weakness',
#     'Stiff Neck':'stiff_neck',
#     'Swelling Joints':'swelling_joints',
#     'Movement Stiffness':'movement_stiffness',
#     'Spinning Movements':'spinning_movements',
#     'Loss Of Balance':'loss_of_balance',
#     'Unsteadiness':'unsteadiness',
#     'Weakness Of One Body Side':'weakness_of_one_body_side',
#     'Loss Of Smell':'loss_of_smell',
#     'Bladder Discomfort':'bladder_discomfort',
#     'Foul Smell Of Urine':'foul_smell_of urine',
#     'Continuous Feel Of Urine':'continuous_feel_of_urine',
#     'Passage Of Gases':'passage_of_gases',
#      'Internal Itching':'internal_itching',
#      'Toxic Look (Typhos)':'toxic_look_(typhos)',
#     'Depression':'depression',
#     'Irritability':'irritability',
#     'Muscle Pain':'muscle_pain',
#     'Altered Sensorium':'altered_sensorium',
#     'Red Spots Over Body':'red_spots_over_body',
#     'Belly Pain':'belly_pain',
#     'Abnormal Menstruation':'abnormal_menstruation',
#     'Dischromic  Patches':'dischromic_patches',
#     'Watering From Eyes':'watering_from_eyes',
#     'Increased Appetite':'increased_appetite',
#     'Polyuria':'polyuria',
#     'Family History':'family_history',
#     'Mucoid Sputum':'mucoid_sputum',
#     'Rusty Sputum':'rusty_sputum',
#     'Lack Of Concentration':'lack_of_concentration',
#     'Visual Disturbances':'visual_disturbances',
#     'Receiving Blood Transfusion':'receiving_blood_transfusion',
#     'Receiving Unsterile Injections':'receiving_unsterile_injections',
#     'Coma':'coma',
#     'Stomach Bleeding':'stomach_bleeding',
#     'Distention Of Abdomen':'distention_of_abdomen',
#     'History Of Alcohol Consumption':'history_of_alcohol_consumption',
#     'Fluid Overload':'fluid_overload',
#     'Blood In Sputum':'blood_in_sputum',
#     'Prominent Veins On Calf':'prominent_veins_on_calf',
#     'Palpitations':'palpitations',
#     'Painful Walking':'painful_walking',
#     'Pus Filled Pimples':'pus_filled_pimples',
#     'Blackheads':'blackheads',
#     'Scurring':'scurring',
#     'Skin Peeling':'skin_peeling',
#     'Silver Like Dusting':'silver_like_dusting',
#     'Small Dents In Nail':'small_dents_in_nails',
#      'Inflammatory Nails':'inflammatory_nails',
#      'Blister':'blister',
#      'Red Sore Around Nose':'red_sore_around_nose',
#      'Yellow Crust Ooze':'yellow_crust_ooze'
# }

def pred(request):
    if request.method=='GET':
        return render(request, 'predic.html')
    else:

        #taking input from user
        symp=request.POST['symptoms']
        symptoms_input=list(symp.split(","))
        output_symp = list(symp.split(","))
        print(symptoms_input)
        temp = []
        final_list=[]
        # for i in range(len(symptoms_input)):
        #     x=symptom_map.get(symptoms_input[i])
        #     final_list.append(x)
        
        # print(final_list)

        for i in range(len(symptoms_input)):
            symptoms_input[i]=symptoms_input[i].strip()
            symptoms_input[i]=symptoms_input[i].lower()
            a=symptoms_input[i].replace(" ","_")
            final_list.append(a)
        print(final_list)
        #process input
        d_name,res,y_pro = get_disease_name(final_list)
        
        print("\nFor symptoms " + str(symptoms_input))
        print("\nDisease predicted: ")
        print(d_name)
        
        print("\nTop 3 disease predictions:")
        summ=sum(res)
        diseases=[]
        prob=[]
        for i in range(len(res)):
            x=np.where(y_pro[0]==res[len(res)-i-1])
            print(i+1,'.',LABELS[x[0][0]],", probability:",round((res[len(res)-i-1]/summ)*100,2),"%")
            diseases.append(LABELS[x[0][0]])
            p=round((res[len(res)-i-1]/summ)*100,2)
            prob.append(p)
        print(diseases)
        print(prob)
        fin = zip(diseases, prob)

      
        
        df=pd.read_csv(os.getcwd()+'../../Dataset/symptom_precaution.csv')
        df2=pd.read_csv(os.getcwd()+'../../Dataset/Symptom-severity.csv')

        res=df[df['Disease']==d_name]
        prec=[]
        prec.append(res['Precaution_1'].values)
        prec.append(res['Precaution_2'].values)
        prec.append(res['Precaution_3'].values)
        prec.append(res['Precaution_4'].values)
        
        flat_prec = [item for sublist in prec for item in sublist]
        flat = [x for x in flat_prec if pd.isnull(x) == False]

        print(flat)

        descr=res['Description'].values
        sev = []
        for i in final_list:
            store  = df2[df2['Symptom']==i]['weight']
            mid = store.tolist()
            sev.append(mid)
        print(sev)
        flat_sev = [item for sublist in sev for item in sublist]
        val = np.mean(flat_sev)
        if(val>=4):
            message = "The symptom severity seems high, Kindly consult a doctor at the earliest."
        else:
            message="The symptom severity is mild as of now. Consult a doctor if remains prolonged."

        return render(request, 'res.html',{'symp': output_symp,'pred':d_name,'diseases':diseases,'prob':prob,'fin':fin, 'prec':flat,'desc':descr,'mess':message,'sev':flat_sev})