import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_gettests(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all tests as JSON format in TestType datatype"
    
    """
    url = f"https://diagnosis.p.rapidapi.com/api/DDxItems/GetTests?AuthenticationID=DEMO_AuthenticationID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diagnosis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_getcategories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all Symptom Categories (Breathing, Muscles, Blood and so on) as JSON format, List of Strings"
    
    """
    url = f"https://diagnosis.p.rapidapi.com/api/DDxItems/GetCategories?AuthenticationID=DEMO_AuthenticationID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diagnosis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_getsymptoms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all Symptoms as JSON format in SymptomType datatype"
    
    """
    url = f"https://diagnosis.p.rapidapi.com/api/DDxItems/GetSymptoms?AuthenticationID=DEMO_AuthenticationID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diagnosis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_getpanels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all panels (groups of tests) as JSON format, List of Strings"
    
    """
    url = f"https://diagnosis.p.rapidapi.com/api/DDxItems/GetPanels?AuthenticationID=DEMO_AuthenticationID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diagnosis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

