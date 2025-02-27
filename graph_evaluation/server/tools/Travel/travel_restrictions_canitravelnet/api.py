import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_supported_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest list of countries that we provide travel restrictions.
		
		The Response is in JSON format."
    
    """
    url = f"https://travel-restrictions-canitravelnet.p.rapidapi.com/api/v1/countries/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-restrictions-canitravelnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_travel_restrictions_for_a_single_country(passport: str, vaccinated: str, country_to_iso: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest personalised travel restrictions for a country based on departure, passport and vaccination.
		
		The Response is in JSON format."
    passport: Optional: 2 letter ISO country code that represents the passport country
        vaccinated: Boolean value: either True or False. By default set to false, true if the traveler has been vaccinated
        from: Optional: 2 letter ISO country code that represents the from country
        
    """
    url = f"https://travel-restrictions-canitravelnet.p.rapidapi.com/api/v1/countries/{country_to_iso}"
    querystring = {'passport': passport, 'vaccinated': vaccinated, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-restrictions-canitravelnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

