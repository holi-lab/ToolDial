import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def statistics(duration: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stats of wheelResults:
		"count"
		"percentage"
		"lastOccurredAt"
		"lastSeenBefore"
		"hotFrequencyPercentage": 1.11
		Note that Duration is Hourly basis 1,2,3,.......72 ect"
    
    """
    url = f"https://immersiverouletteapi.p.rapidapi.com/stats"
    querystring = {}
    if duration:
        querystring['duration'] = duration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immersiverouletteapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_outcome(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Last Spin data"
    
    """
    url = f"https://immersiverouletteapi.p.rapidapi.com/latest-outcome"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immersiverouletteapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the last 20 spins histories includes all details about the game"
    
    """
    url = f"https://immersiverouletteapi.p.rapidapi.com/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immersiverouletteapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

