import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_an_overview_for_a_summoner(region: str, summonersname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns an overview of the summoner from U.GG
		
		For the region you have to choose between:
		"na1" , "euw1", "eun1" , "kr",  "br1" ,"jp1", "ru" , "oc1" , "tr1", "la1" , "la2""
    
    """
    url = f"https://lol_stats.p.rapidapi.com/{region}/{summonersname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lol_stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

