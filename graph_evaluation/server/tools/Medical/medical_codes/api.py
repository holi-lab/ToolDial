import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2_icd_10_cm_code_summary(icdcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the basic summary for any ICD-10-CM code."
    
    """
    url = f"https://medical-codes.p.rapidapi.com/code/{icdcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_icd_10_cm_code_details(icdcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup the summary about an ICD-10-CM code."
    
    """
    url = f"https://medical-codes.p.rapidapi.com/code/{icdcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4_icd_10_cm_diagnosis_details(icdcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup ICD-10-CM diagnosis details"
    
    """
    url = f"https://medical-codes.p.rapidapi.com/diagnosis/{icdcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_icd_10_cm_chapter_details(chapter: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup the ICD-10-CM chapter details and information."
    
    """
    url = f"https://medical-codes.p.rapidapi.com/chapter/{chapter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_icd_10_cm_section_lookup(sectionrange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a section"
    
    """
    url = f"https://medical-codes.p.rapidapi.com/section/{sectionrange}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_icd_10_cm_main_term_lookup(mainterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a main term from the icd-10-cm index."
    
    """
    url = f"https://medical-codes.p.rapidapi.com/term/{mainterm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_icd_10_cm_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for ICD-10-CM codes using a phrase, keyword, diagnosis, icd code pattern,  or icd code
		
		
		#### Example search queries that you can try.
		Hypertrophy of the tonsils
		Recurrent ventral hernia 
		Urinary tract infection from e coli
		T38 with fifth character 1-2 or 4-5
		V05-V06 with 5th character 1
		E08-E13 with .21
		S37.05-
		P04-P96
		W61.61XD
		Y92.010
		X99.1"
    
    """
    url = f"https://medical-codes.p.rapidapi.com/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medical-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

