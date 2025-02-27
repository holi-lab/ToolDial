import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def indonesian_disaster_endpoint_by_month_and_year(year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indonesian Disaster Endpoint Filter By Month And Year"
    
    """
    url = f"https://indonesia-disaster-data.p.rapidapi.com/bnpb_incident"
    querystring = {'year': year, 'month': month, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-disaster-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

