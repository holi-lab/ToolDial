import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_champions(champions: str='cait', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts a comma delimited list of champion names or nicknames.
		
		Returns a list of Champions full names and list of passive ability and standard abilities in an on object format."
    
    """
    url = f"https://league-of-legends-champion-informaion.p.rapidapi.com/LolInfo/GetChampions"
    querystring = {}
    if champions:
        querystring['champions'] = champions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-informaion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_abilities_as_text(champion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts a comma delimited list of champion names/nicknames and returns a text list of champions as well as their passive and standard abilities."
    
    """
    url = f"https://league-of-legends-champion-informaion.p.rapidapi.com/LolInfo/GetChampionAbilitiesText"
    querystring = {'champion': champion, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-informaion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

