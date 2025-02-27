import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def singapore_clinics(x_api_key: str='x7fS9ajAKm8AhmrimAZ4u5lUnrK5GZZZ48QlPIYu', doctor: str='Chao Wen Pin Cynthia', name_like: str='boon', name: str='BOON LAY CLINIC & SURGERY', doctor_like: str='Cynthia', q: str='boon', lisence: str='9400129', limit: str='5', address_like: str='boon lay', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Singapore Clinics
		
		Most of the parameters should be the same as per[ this github.](https://github.com/typicode/json-server)"
    doctor: Find exact match, even in capitalisation.
        name_like: Add `_like` to name to filter clinics with name similar to parameter.

Accept Regex
        name: use `name` to match the exact name of the clinic in **full capital letters.**
        doctor_like: Find doctors with name similar to search term.
        q: Full text search
        lisence: Search for exact match in lisence number
        limit: Used to limit search results
        address_like: Search for address that contains parameter.
        
    """
    url = f"https://singapore-clinics.p.rapidapi.com/clinics"
    querystring = {}
    if x_api_key:
        querystring['x-api-key'] = x_api_key
    if doctor:
        querystring['doctor'] = doctor
    if name_like:
        querystring['name_like'] = name_like
    if name:
        querystring['name'] = name
    if doctor_like:
        querystring['doctor_like'] = doctor_like
    if q:
        querystring['q'] = q
    if lisence:
        querystring['lisence'] = lisence
    if limit:
        querystring['_limit'] = limit
    if address_like:
        querystring['address_like'] = address_like
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "singapore-clinics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

