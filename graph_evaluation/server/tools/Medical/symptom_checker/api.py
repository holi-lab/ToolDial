import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def issues(issues: str=None, language: str='en-gb', format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Issues can be either called to receive the full list of issues or a subset of issues (e.g. all issues of a diagnosis)."
    issues: JSON encoded int[] array	Serialized array of selected issue ids in json format. example issues=[234,235,236]
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/issues"
    querystring = {}
    if issues:
        querystring['issues'] = issues
    if language:
        querystring['language'] = language
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symptoms_in_body_sublocations(locationid: int, language: str, selectorstatus: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Symptoms in body sublocations can be called to receive all the symptoms in a body sub location. It returns an array of all symptoms in a body sublocations for the requested locationId. Each element consists of the ID, Name, HasRedFlag, HealthSymptomLocationIDs and Synonyms.  If locationId=0, then you get all symptoms (from all body locations)  selectorStatus has been given the values “man, woman, boy, girl” (MWBG). The switch from Child to Adult is at the age of 11 (“edge year”). This means that 12 and above is considered as Adult (for man, woman), and 11 and below is considered as Child (for boy, girl). Meaning whoever indicates the age 11 and below (2016 - 2005 = 11) is considered a child (so Boy or Girl) and whoever indicates the age 12 and above (2016 - 2004 = 12) is considered adult."
    locationid: see Body sublocations. If locationId = 0, then you get all symptoms
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        selectorstatus: Options are man, woman, boy, girl. The switch from Child to Adult is at the age of 11 (“edge year”). This means that 12 and above is considered as Adult (for man, woman), and 11 and below is considered as Child (for boy, girl)
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/symptoms/{locationid}/{selectorstatus}"
    querystring = {'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_sublocations(locationid: str, language: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Body sublocations can be called to receive all the body sub locations from a body location."
    locationid: Body sublocations can be called to receive all the body sub locations from a body location. It returns an array of all body sublocations for the requested location_id. Each element consists of the ID and Name.
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/body/locations/{locationid}"
    querystring = {'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symptoms(language: str, format: str='json', symptoms: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Symptoms can be either called to receive the full list of symptoms or a subset of symptoms (e.g. all symptoms of a body sublocation)."
    language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        symptoms: Serialized array of selected symptom ids in json format. example symptoms=[234,11] (JSON encoded int[] array)
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/symptoms"
    querystring = {'language': language, }
    if format:
        querystring['format'] = format
    if symptoms:
        querystring['symptoms'] = symptoms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diagnosis(gender: str, year_of_birth: int, symptoms: str, language: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The diagnosis is the core function of the symptom-checker to compute the potential health issues based on a set of symptoms, gender and age."
    gender: Options: male or female
        year_of_birth: Year of birth of the patient.
        symptoms: Serialized array of selected symptom ids as a JSON encoded int[] array. Example: [234,235,236]
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/diagnosis"
    querystring = {'gender': gender, 'year_of_birth': year_of_birth, 'symptoms': symptoms, 'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_locations(language: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Body locations can be called to receive all the body locations"
    language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/body/locations"
    querystring = {'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def issue_info(language: str, issue_id: int, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Issue info can be called to receive all information about a health issue. The short description gives a short overview. A longer information can consist of "Description", "MedicalCondition", "TreatmentDescription"."
    language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        issue_id: Number of the health issue
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/issues/{issue_id}/info"
    querystring = {'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def proposed_symptoms(gender: str, year_of_birth: int, language: str, symptoms: str='[234,11]', format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The proposed symptoms can be called to request additional symptoms which can be related to the given symptoms in order to refine the diagnosis."
    gender: Male or female
        year_of_birth: Year of birth
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        symptoms: Serialized array of selected symptom ids in json format. example symptoms=[234,235,236]
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/symptoms/proposed"
    querystring = {'gender': gender, 'year_of_birth': year_of_birth, 'language': language, }
    if symptoms:
        querystring['symptoms'] = symptoms
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def red_flag_text(language: str, symptomid: int=17, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Red flag texts are recommendations to the patient for a higher urgency or severeness of the possible symptoms. As an example a patient with pain in the breast might have a heart attack and therefore the patient should be warned about the urgency and severeness of the matter."
    language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        symptomid: Red flag texts are recommendations to the patient for a higher urgency or severeness of the possible symptoms. As an example a patient with pain in the breast might have a heart attack and therefore the patient should be warned about the urgency and severeness of the matter.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/redflag"
    querystring = {'language': language, }
    if symptomid:
        querystring['symptomId'] = symptomid
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specialisations_based_on_diagnosis(symptoms: str, gender: str, year_of_birth: int, language: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The diagnosis is the core function of the symptom-checker to compute the potential health issues based on a set of symptoms, gender and age, but instead of getting computed diagnosis, you can also get list of suggested specialisations for calulated diseases"
    symptoms: Serialized array of selected symptom ids in json format. example symptoms=[234,235,236]
        gender: Male or female
        year_of_birth: Year of birth
        language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        format: Response format: json or xml.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/diagnosis/specialisations"
    querystring = {'symptoms': symptoms, 'gender': gender, 'year_of_birth': year_of_birth, 'language': language, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specialisations_list(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check out our specialisations so you can make your own mapping."
    language: Define language flag for information to be retrieved. Currently possible: de-ch, en-gb, fr-fr, es-es, tr-tr.
        
    """
    url = f"https://priaid-symptom-checker-v1.p.rapidapi.com/specialisations"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priaid-symptom-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

