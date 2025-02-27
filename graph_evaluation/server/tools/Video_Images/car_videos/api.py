import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video(manufacturerid: int=1, offset: int=0, limit: int=10, modelid: int=6, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can find tons of videos and filter them by modelId and manufacturerId"
    
    """
    url = f"https://car-videos.p.rapidapi.com/video"
    querystring = {}
    if manufacturerid:
        querystring['manufacturerId'] = manufacturerid
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if modelid:
        querystring['modelId'] = modelid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manufacturers_models(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all cars manufacturers with all models they have."
    
    """
    url = f"https://car-videos.p.rapidapi.com/manufacturers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

