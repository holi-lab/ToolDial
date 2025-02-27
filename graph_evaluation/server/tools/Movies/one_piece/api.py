import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_character_by_name(token: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Character by Name"
    
    """
    url = f"https://one-piece2.p.rapidapi.com/v2/getCharacter/{name}"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "one-piece2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_character_by_devil_fruit_type(dftype: str, token: str='ab84ad27eb9fe47b625069a7f0a4833fb92439639d9a57f7a56ca60bc4a8fbc6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Paramecia
		- Logia
		- Zoan"
    
    """
    url = f"https://one-piece2.p.rapidapi.com/v2/getCharacterbyDF/{dftype}"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "one-piece2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_episode_info(token: str, episode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Episode Info with Episode Number"
    
    """
    url = f"https://one-piece2.p.rapidapi.com/getEpisode/{episode}"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "one-piece2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_characters(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Characters"
    
    """
    url = f"https://one-piece2.p.rapidapi.com/v2/getAllCharacters"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "one-piece2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

