import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getproducts(query: str, page: int, country: str='countryUS', location: str='us', lang: str='en', period: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products by name and retrieve newly added items from the last X days, among other features."
    
    """
    url = f"https://get-and-compate-products-allover-the-web.p.rapidapi.com/"
    querystring = {'query': query, 'page': page, }
    if country:
        querystring['country'] = country
    if location:
        querystring['location'] = location
    if lang:
        querystring['lang'] = lang
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-and-compate-products-allover-the-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

