import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this end point will fetch the top shower thought from all frames of time. 
		(i.e. now, today,week,month,year,alltime)"
    
    """
    url = f"https://stapi-showerthoughts.p.rapidapi.com/api/v1/stapi/top"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stapi-showerthoughts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest(num: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "chose a number between 1 to 500 and based on that all the latest showerthoughts on the subReddit will be fetched.
		
		if the number is not specified a total of 100 showerthougts are returned by default."
    
    """
    url = f"https://stapi-showerthoughts.p.rapidapi.com/api/v1/stapi/latest"
    querystring = {}
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stapi-showerthoughts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches a random shower thought out of latest 100 posted on the subReddit."
    
    """
    url = f"https://stapi-showerthoughts.p.rapidapi.com/api/v1/stapi/randomnew"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stapi-showerthoughts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

