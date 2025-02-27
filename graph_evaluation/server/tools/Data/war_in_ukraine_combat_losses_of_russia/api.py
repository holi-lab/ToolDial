import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getallstatisticalterms(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for selecting information about statistical terms."
    lang: Translate statistics fields titles in response to the selected language.
        
    """
    url = f"https://war-in-ukraine-combat-losses-of-russia.p.rapidapi.com/terms/{lang}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "war-in-ukraine-combat-losses-of-russia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatisticalterm(lang: str, term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for selecting information about statistical term."
    lang: Translate statistics fields titles in response to the selected language.
        term: Name of statistics term.
        
    """
    url = f"https://war-in-ukraine-combat-losses-of-russia.p.rapidapi.com/terms/{lang}/{term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "war-in-ukraine-combat-losses-of-russia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlateststatistic(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for selecting information about the latest statistic record with increase."
    
    """
    url = f"https://war-in-ukraine-combat-losses-of-russia.p.rapidapi.com/statistics/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "war-in-ukraine-combat-losses-of-russia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallstatistics(limit: int=1, offset: int=49, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for selecting basic information about all statistics."
    limit: The numbers of items to return. The maximum value is 50.
        offset: The number of items to skip before starting to collect the result set. The minimum value is 0.
        
    """
    url = f"https://war-in-ukraine-combat-losses-of-russia.p.rapidapi.com/statistics"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "war-in-ukraine-combat-losses-of-russia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatisticbydate(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for selecting basic information about a statistic by date."
    date: Date string in "YYYY-MM-DD" format, reflecting the day of war.
        
    """
    url = f"https://war-in-ukraine-combat-losses-of-russia.p.rapidapi.com/statistics/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "war-in-ukraine-combat-losses-of-russia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

