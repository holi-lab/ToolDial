import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_airport(limit: int, keyword: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint which is paginated also could be used to search a particular record by airport keyword or location keyword. This could be applicable when creating auto-completes etc."
    
    """
    url = f"https://aviation-airports-and-heliports-api.p.rapidapi.com/api/v1/aviationsrvc/airports/search"
    querystring = {'limit': limit, 'keyword': keyword, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-airports-and-heliports-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_airports(page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint which is paginated gets all airports, airfields, heliports etc."
    
    """
    url = f"https://aviation-airports-and-heliports-api.p.rapidapi.com/api/v1/aviationsrvc/airports/all"
    querystring = {'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-airports-and-heliports-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

