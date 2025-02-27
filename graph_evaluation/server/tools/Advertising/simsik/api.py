import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def genders_by_names(name1: str, name2: str, country2: str='uk', country1: str='us', language1: str='en', language2: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get genders by names - bulk request. Max request size is 50 names."
    name1: first name
        name2: second name
        country2: second name country code
        country1: first name country code
        language1: first name language code
        language2: second person language code
        
    """
    url = f"https://name-gender.p.rapidapi.com/getGenders"
    querystring = {'name1': name1, 'name2': name2, }
    if country2:
        querystring['country2'] = country2
    if country1:
        querystring['country1'] = country1
    if language1:
        querystring['language1'] = language1
    if language2:
        querystring['language2'] = language2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-gender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gender_by_name(name: str, country: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gender by name"
    name: name
        country: country code
        language: language code
        
    """
    url = f"https://name-gender.p.rapidapi.com/getGender"
    querystring = {'name': name, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-gender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

