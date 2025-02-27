import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def icd_10_in_italian(malattia: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve codification on **CID-10** for a specific disease or condition in Italian."
    
    """
    url = f"https://international-disease-classification-10th-edition-icd10.p.rapidapi.com/icd_italiano"
    querystring = {'malattia': malattia, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-disease-classification-10th-edition-icd10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icd_10_in_english(disease: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a Json describing a list of diseases and their ICD-10 code according to the search terms."
    
    """
    url = f"https://international-disease-classification-10th-edition-icd10.p.rapidapi.com/icd"
    querystring = {'disease': disease, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-disease-classification-10th-edition-icd10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cid_10_in_portuguese(disease: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve codification on **CID-10** for a specific disease or condition in Portuguese."
    
    """
    url = f"https://international-disease-classification-10th-edition-icd10.p.rapidapi.com/cid"
    querystring = {'disease': disease, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-disease-classification-10th-edition-icd10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

