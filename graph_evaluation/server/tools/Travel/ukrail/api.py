import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getukrail(cmd: str, numberqueries: int=5, crs: str='KGX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query the API to get rail data."
    
    """
    url = f"https://ukrail.p.rapidapi.com/GetUKRail"
    querystring = {'CMD': cmd, }
    if numberqueries:
        querystring['NumberQueries'] = numberqueries
    if crs:
        querystring['CRS'] = crs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ukrail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

