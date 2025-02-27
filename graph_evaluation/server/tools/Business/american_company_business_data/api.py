import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(p: int, ls: int, c: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search business contacts"
    p: Search by: 1 - company name, 2- address, 3-email, 4-web, 5-industry, 6-zip
        ls: Limit Start Page Offset: 0,50,100,150,...N
        c: Country Selector. Available Options: US, GB
        s: Search Query. Example: bank, sport, music ... etc
        
    """
    url = f"https://american-company-business-data.p.rapidapi.com/"
    querystring = {'p': p, 'ls': ls, 'c': c, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-company-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

