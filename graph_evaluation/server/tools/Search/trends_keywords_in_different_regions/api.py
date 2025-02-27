import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def last10(geo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This getting the last 10 day trending keyword of geo location"
    
    """
    url = f"https://trends-keywords-in-different-regions.p.rapidapi.com/last10"
    querystring = {'geo': geo, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trends-keywords-in-different-regions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trend_keyword(date: str='2023-05-18', geo: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API provides the flexibility to filter trending keywords based on a particular date, ensuring that developers receive precise and relevant information. By specifying the date as '2023-05-16', developers can retrieve trending keywords specifically for that day, enabling them to analyze and incorporate the most recent trends into their applications."
    
    """
    url = f"https://trends-keywords-in-different-regions.p.rapidapi.com/daily"
    querystring = {}
    if date:
        querystring['date'] = date
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trends-keywords-in-different-regions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_geo_map_for_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Trending Keywords by Geo Regions" API allows developers to access a list of 50 geographic regions and their corresponding trending keywords. With this API, developers can retrieve valuable insights into the most popular and trending search terms for specific locations."
    
    """
    url = f"https://trends-keywords-in-different-regions.p.rapidapi.com/geomap"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trends-keywords-in-different-regions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

