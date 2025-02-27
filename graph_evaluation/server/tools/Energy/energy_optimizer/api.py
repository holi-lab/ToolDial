import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_hours_available(count: int, country: str, start_time: str=None, end_time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Returns** : json of chepaest hours info, for future time interval
		Parameters: 
		**country** - one of supported 2-letter formats
		**count** - number of hours to return
		**start_time** - ISO formatted with time offset, like 2022-11-28T18:00:00+02:00, default next hour
		**end_time** - ISO formatted with time offset, like 2022-11-28T18:00:00+02:00, default end of available prices"
    
    """
    url = f"https://energy-optimizer.p.rapidapi.com/{country}/hours_available/{count}"
    querystring = {}
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "energy-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hours(count: int, country: str, period: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Returns** : json of chepaest hours, for full periods (past time included), depending period parameter
		Parameters:
		**country** - one of supported 2-letter formats
		**count** - number of hours to return
		**period** - may be one of 0 - today (default), 1 - tomorrow, 2 - today and tomorrow"
    
    """
    url = f"https://energy-optimizer.p.rapidapi.com/{country}/hours/{count}"
    querystring = {}
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "energy-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return version and data last updated info"
    
    """
    url = f"https://energy-optimizer.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "energy-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

