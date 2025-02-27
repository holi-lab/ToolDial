import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_01_check_person(last_name: str, first_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Structure of the query URL for a person:
		  https://index.smartapicloud.com/sl/<first name>/<last name>/
		
		Structure of the json result string:
		
		  {'result': True/False, 'data': {'firstname': 'data', 'lastname': 'data', 'wholename': 'data', 'born': 'data', 'country': 'data', 'city': 'data', 'zip': 'data', 'street': 'data', 'publication': 'data'}}"
    
    """
    url = f"https://consolidated-financial-sanctions-list-european-union.p.rapidapi.com/{first_name}/{last_name}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consolidated-financial-sanctions-list-european-union.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_02_check_company(company_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Structure of the query URL for a company:
		  https://index.smartapicloud.com/sl/<company name>/
		
		Structure of the json result string:
		
		  {'result': True/False, 'data': {'firstname': 'data', 'lastname': 'data', 'wholename': 'data', 'born': 'data', 'country': 'data', 'city': 'data', 'zip': 'data', 'street': 'data', 'publication': 'data'}}"
    
    """
    url = f"https://consolidated-financial-sanctions-list-european-union.p.rapidapi.com/{company_name}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consolidated-financial-sanctions-list-european-union.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

