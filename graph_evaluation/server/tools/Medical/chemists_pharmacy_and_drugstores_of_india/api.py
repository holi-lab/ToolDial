import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dentist_api(page_no: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of dentists and doctors of Jaipur."
    
    """
    url = f"https://chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com/dentist/{page_no}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doctors_api(page_no: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get doctors details  of india."
    
    """
    url = f"https://chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com/doctors/{page_no}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chemist_api(page_no: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of almost 82K chemists including their office time."
    page_no: Get the details of almost 84K chemists including their office time.
        
    """
    url = f"https://chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com/chemist/{page_no}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chemists-pharmacy-and-drugstores-of-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

