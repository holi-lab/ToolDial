import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def raiderio_call(region: str, realm: str, fields: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calls the Raider.io API to get info"
    region: Region (us)
        realm: Realm (stormrage)
        fields: Field (gear or mythic_plus_scores_by_season:current
        name: Character Name
        
    """
    url = f"https://raider-io.p.rapidapi.com/api/v1/characters/profile"
    querystring = {'region': region, 'realm': realm, 'fields': fields, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raider-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

