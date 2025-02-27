import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def celebrity_birthdays(format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Celebrity's birthday"
    format: JSON or XML
        
    """
    url = f"https://celebrity-bucks.p.rapidapi.com/birthdays/{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "celebrity-bucks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_celebrity_pricing_xml(format: str, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The BonusBucks node will provide any celebrities that are currently designated as "Bonus Bucks" celebrities in the system. These are temporary price changes and will expire on the date/time in the expires node.  The CelebrityValues node will provide all celebrities who currently have a price in the Celebrity Bucks Exchange."
    format: JSON or XML
        limit: If you only need a certain number of results (Top 10 for example), add a GET variable of \"limit\".  For example, the URL for the Top 10 celebrities in XML format would be https://celebritybucks.com/developers/export/XML?limit=10.
        
    """
    url = f"https://celebrity-bucks.p.rapidapi.com/export/{format}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "celebrity-bucks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

