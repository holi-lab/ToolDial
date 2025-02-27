import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index_wilaya(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Weather in specific wilaya with index wilaya
		for ex /wilaya/16 
		this end point is for Wilaya Alger
		for ex /wilaya/38
		this endpoint is for Wilaya Tissemsilt"
    
    """
    url = f"https://dz-meteo-live.p.rapidapi.com/wilaya/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dz-meteo-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_wilayas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all wilayas weather,
		with: 
		index_wilaya
		name
		temp
		statue
		img"
    
    """
    url = f"https://dz-meteo-live.p.rapidapi.com/allwilayas"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dz-meteo-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

