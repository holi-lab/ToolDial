import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_character_by_name(charactername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By inputting the name as it appears on the Wiki, would return information pertaining to Character."
    
    """
    url = f"https://overlord-characters.p.rapidapi.com/characters/name/{charactername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "overlord-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_400_characters_unorganized(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Entire List of Characters from Overlord up to Volume 14."
    
    """
    url = f"https://overlord-characters.p.rapidapi.com/characters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "overlord-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_characters_by_affiliation(affiliationname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Characters in Overlord have various Affiliations they they recognize themselves under ie "Blue Roses",  and "Darkness", by putting in the organization will return the Characters that belong to it."
    
    """
    url = f"https://overlord-characters.p.rapidapi.com/characters/affiliation/{affiliationname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "overlord-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

