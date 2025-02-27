import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_mountain_peak_by_name(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search for mountain peaks by name. returns all mountains with names containing the search query, along with their associated peak_id."
    query: The search query - enter the name of the mountain peak you wish to search for. spaces can be represented by the '+' character.
        
    """
    url = f"https://peak-conditions.p.rapidapi.com/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "peak-conditions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_report_by_mountain_id(mountainid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a daily report for the mountain peak specified in the request."
    mountainid: The id number associated with a mountain peak. 
        
    """
    url = f"https://peak-conditions.p.rapidapi.com/report/daily/{mountainid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "peak-conditions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_extended_report_by_mountain_id(mountainid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an extended 6-day report for the mountain peak specified in the request."
    
    """
    url = f"https://peak-conditions.p.rapidapi.com/report/extended/{mountainid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "peak-conditions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

