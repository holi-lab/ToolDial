import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_license_plate(state: str, plate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by license plate number for the owning company details, inspections, crashes and other valuable data"
    
    """
    url = f"https://motor-carrier-search.p.rapidapi.com/search_plate"
    querystring = {'State': state, 'Plate': plate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motor-carrier-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def legal_name_search(legal_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for details by company name"
    
    """
    url = f"https://motor-carrier-search.p.rapidapi.com/search_legalname"
    querystring = {'LEGAL_NAME': legal_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motor-carrier-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dot_number_search(dot_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter a DOT Number for search"
    
    """
    url = f"https://motor-carrier-search.p.rapidapi.com/search_dot"
    querystring = {'DOT_NUMBER': dot_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motor-carrier-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_vin(vin_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by vehicle identification number for the owning company details, inspections, crashes and other valuable data"
    
    """
    url = f"https://motor-carrier-search.p.rapidapi.com/search_vinnumber"
    querystring = {'VIN_NUMBER': vin_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motor-carrier-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

