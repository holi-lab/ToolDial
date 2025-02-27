import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def diseasemedications(country: str, medius_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup medications that are used to treat a particular disease or condition.
		
		##Parameters
		| Name      | Type   | Description                                                                                                                                                            | Optional |
		|-----------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
		| medius_id | string | Identifies the disease in the Medius Health knowledge base.                                                                                                            | N        |
		| country   | string | Currently only supports 'IN' (India) | N        |"
    
    """
    url = f"https://medius-disease-medication.p.rapidapi.com/api/v2/disease-medications/{medius_id}"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medius-disease-medication.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diseasesearch(query: str, age: int=27, birth_sex: str='UNK', specialization: str='general_practitioner', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint facilitates matching of a free text disease name with diseases in Medius's knowledge base.
		
		##Parameters
		| name           | type    | description                                                                                                                         | Optional |
		|----------------|---------|-------------------------------------------------------------------------------------------------------------------------------------|----------|
		| query          | string  | The free text to be searched                                                                                                        | N        |
		| birth_sex      | string  | The patient’s sex at birth. ‘M' for male, ‘F’ for female and ‘UNK’ for unknown.                                                     | Y        |
		| age            | integer | The patient’s age in years.                                                                                                         | Y        |
		| specialization | string  | A code that uniquely identifies the area of specialization on which the results are to be constrained to (see list of codes below). | Y        |
		
		## Specialization codes
		| Code                                 | Description                                                                                                             |
		|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
		| allergy\_immunology\_doctor            | Treats allergy and immune system disorders.                                                                             |
		| anaesthesiologist                    | Administers drugs to manage chronic pain or make a patient unaware of pain during a medical procedure.                  |
		| audiologist                          | Treats hearing loss and balance disorders.                                                                              |
		| cardiac_electrophysiologist          | Treats patients with electrical and heart rhythm problems.                                                              |
		| cardiologist                         | Specialises in heart disorders.                                                                                         |
		| cardiothoracic_surgeon               | Surgically treats diseases affecting organs inside the chest.                                                           |
		| chiropodist                          | Treats disorders of the foot.                                                                                           |
		| clinical_psychologist                | Treats mental disorders primarily with talk therapy.                                                                    |
		| colon\_rectal\_surgeon                 | Performs surgery on the intestinal tract.                                                                               |
		| critical\_care\_doctor                 | Monitors and treats those in intensive care.                                                                            |
		| dentist                              | Specialises in diseases of the oral cavity, especially the teeth.                                                       |
		| dermatologist                        | Focuses on disorders of skin, nails and hair.                                                                           |
		| emergency\_medicine\_doctor            | Treats patients in the emergency department.                                                                            |
		| endocrinologist                      | Treats metabolic and hormone disorders.                                                                                 |
		| gastroenterologist                   | Focuses on the digestive system and its disorders.                                                                      |
		| general_practitioner                 | Prevents, diagnoses and treats diseases.                                                                                |
		| general_surgeon                      | Performs a range of surgeries on the abdomen, skin, breast and soft tissue.                                             |
		| geriatrician                         | Focuses on the health care of elderly people.                                                                           |
		| haematologist                        | Focuses on diagnosing and treating blood disorders.                                                                     |
		| hand_surgeon                         | Performs surgery to treat hand conditions.                                                                              |
		| hepatologist                         | Focuses on liver, gallbladder and biliary tree disorders.                                                               |
		| immunologist                         | Diagnoses and treats immune system disorders.                                                                           |
		| infectious\_disease\_doctor            | Treats infections, including those that are tropical in nature.                                                         |
		| internal\_medicine\_physician          | Provides the diagnosis, treatment, and compassionate care of adults across the spectrum from health to complex illness. |
		| interventional_radiologist           | Uses image-guided procedures to diagnose and treat diseases.                                                            |
		| maternal\_foetal\_medicine\_specialist  | Focuses on the medical management of high-risk pregnancies.                                                             |
		| medical_geneticist                   | Diagnoses and manages hereditary disorders.                                                                             |
		| nephrologist                         | Focuses on kidney disorders.                                                                                            |
		| neurologist                          | Treats nervous system disorders.                                                                                        |
		| neurosurgeon                         | Specialises in nervous system disorders.                                                                                |
		| nutritionist                         | Specialises in food and diet.                                                                                           |
		| obgyn_doctor                         | Focuses on reproductive health in women and childbirth.                                                                 |
		| occupational_therapist               | Improves daily living and work skills of patients.                                                                      |
		| oncologist                           | Specialises in cancer.                                                                                                  |
		| ophthalmologist                      | Specialises in eye diseases.                                                                                            |
		| optometrist                          | Diagnoses and treats vision changes.                                                                                    |
		| orthopaedic_surgeon                  | Performs surgery for conditions affecting bones and muscles.                                                            |
		| otolaryngologist                     | Treats ear, nose and throat disorders.                                                                                  |
		| otorhinolaryngologist                | Treats ear, nose and throat disorders.                                                                                  |
		| paediatric_dermatologist             | Focuses on childhood disorders of skin, nails and hair.                                                                 |
		| paediatric_neurologist               | Treats nervous system disorders in children.                                                                            |
		| paediatric_surgeon                   | Performs surgery in children.                                                                                           |
		| paediatrician                        | Provides medical care for infants, children and teenagers.                                                              |
		| pain_management                      | Eases suffering and improves quality of life for those in pain.                                                         |
		| physical\_medicine\_and\_rehabilitation | Restores function and quality of life to those with physical disabilities.                                              |
		| physiotherapist                      | Restores muscle strength and function through exercise.                                                                 |
		| plastic_surgeon                      | Reconstructs defective, damaged or missing body parts.                                                                  |
		| psychiatrist                         | Treats mental disorders primarily with medications.                                                                     |
		| pulmonologist                        | Treats respiratory tract diseases.                                                                                      |
		| radiation_oncologist                 | Treats and manages cancer by prescribing radiation therapy.                                                             |
		| respiratory_therapist                | Treats and provides emergency care for patients who have trouble breathing.                                             |
		| rheumatologist                       | Specialises in arthritis and other rheumatic diseases.                                                                  |
		| sleep_medicine                       | Treats sleep disturbances and disorders.                                                                                |
		| speech_therapist                     | Treats people with speech and language problems.                                                                        |
		| sports_medicine                      | Treats and prevents sports and exercise injuries.                                                                       |
		| surgeon                              | Performs operations to treat disease.                                                                                   |
		| transplant_surgeon                   | Transfers an organ from one body to another.                                                                            |
		| travel_medicine                      | Manages health concerns for international travellers.                                                                   |
		| urologist                            | Treats urinary tract diseases.                                                                                          |
		| vascular_surgeon                     | Specialises in surgery for vascular system diseases.                                                                    |"
    
    """
    url = f"https://medius-disease-medication.p.rapidapi.com/api/v2/disease-search"
    querystring = {'query': query, }
    if age:
        querystring['age'] = age
    if birth_sex:
        querystring['birth_sex'] = birth_sex
    if specialization:
        querystring['specialization'] = specialization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medius-disease-medication.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

