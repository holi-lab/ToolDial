import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def practice_search(name: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for provider service locations (practices or doctor's offices)."
    name: The practice’s name. Partial (type-ahead) names are accepted.
        location: Search area - either circular (lat,lon,range (miles)) or bounding box (top-right, bottom-left) (lat,lon,lat,lon). Location slugs (ca-berkeley) are also accepted. Slugs are returned as part of the doctor's practices. Also supports state codes e.g. NY or ny.
        
    """
    url = f"https://betterdoctor.p.rapidapi.com/api.betterdoctor.com/2016-03-01/practices"
    querystring = {}
    if name:
        querystring['Name'] = name
    if location:
        querystring['Location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betterdoctor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doctor_search_by_name(name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for doctors using name."
    name: The doctor's name; searches in both first and last names. Partial (type-ahead) characters/names are accepted.
        
    """
    url = f"https://betterdoctor.p.rapidapi.com/api.betterdoctor.com/2016-03-01/doctors?name="
    querystring = {}
    if name:
        querystring['Name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betterdoctor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

