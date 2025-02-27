import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_u_58_athlete_ranking(country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://taekwondo_athlete_world_ranking1.p.rapidapi.com/GET_U-58_ATHLETE_RANKING"
    querystring = {}
    if country:
        querystring['Country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taekwondo_athlete_world_ranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_u_54_athlete_ranking(country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://taekwondo_athlete_world_ranking1.p.rapidapi.com/GET_U-54_ATHLETE_RANKING"
    querystring = {}
    if country:
        querystring['Country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taekwondo_athlete_world_ranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

