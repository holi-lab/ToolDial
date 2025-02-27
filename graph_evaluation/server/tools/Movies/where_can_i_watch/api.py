import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(country: str, title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for streaming options for movies and series in various countries"
    country: Examples:
uk, dk, us, in
        
    """
    url = f"https://where-can-i-watch1.p.rapidapi.com/search/{country}/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "where-can-i-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_title(type: str, title: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup streaming options for a specific movie or series in various countries"
    type: Type is either movie or series
        title: If you do not get any results try and put spaces between each letter like

M A S H
        country: Examples:
uk, dk, us, in
        
    """
    url = f"https://where-can-i-watch1.p.rapidapi.com/{country}/{type}/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "where-can-i-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

