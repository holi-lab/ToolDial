import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def holidays(optional: str='false', federal: str='1', year: int=2023, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Canadian public holidays. Each holiday lists the regions that observe it."
    optional: A boolean parameter. If false or 0 (default), will return only legislated holidays. If true or 1, will return optional holidays from Alberta and BC.
        federal: A boolean parameter. If true or 1, will return only federal holidays. If false or 0, will return no federal holidays.
        year: A calendar year
        
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1/holidays"
    querystring = {}
    if optional:
        querystring['optional'] = optional
    if federal:
        querystring['federal'] = federal
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holiday(holidayid: int, year: int=2023, optional: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns one Canadian statutory holiday by integer id. Returns a 404 response for invalid ids."
    holidayid: Primary key for a holiday
        year: A calendar year
        optional: A boolean parameter. If false or 0 (default), will return provinces for which this is a legislated holiday. If true or 1, will return provinces which optionally celebrate this holiday.
        
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1/holidays/{holidayid}"
    querystring = {}
    if year:
        querystring['year'] = year
    if optional:
        querystring['optional'] = optional
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def province(provinceid: str, year: int=2023, optional: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Canadian province or territory with its associated holidays. Returns a 404 response for invalid abbreviations."
    provinceid: A Canadian province abbreviation
        year: A calendar year
        optional: A boolean parameter (AB and BC only). If false or 0 (default), will return only legislated holidays. If true or 1, will return optional holidays from Alberta and BC.
        
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1/provinces/{provinceid}"
    querystring = {}
    if year:
        querystring['year'] = year
    if optional:
        querystring['optional'] = optional
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def root(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a welcome message."
    
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def provinces(optional: str='false', year: int=2023, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns provinces and territories in Canada. Each province or territory lists its associated holidays."
    optional: A boolean parameter. If false or 0 (default), will return only legislated holidays. If true or 1, will return optional holidays from Alberta and BC.
        year: A calendar year
        
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1/provinces"
    querystring = {}
    if optional:
        querystring['optional'] = optional
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spec(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the schema for the JSON API as a yaml file."
    
    """
    url = f"https://canada-holidays.p.rapidapi.com/api/v1/spec"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canada-holidays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

