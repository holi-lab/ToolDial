import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_billionaires(page: str, size: str, year: str, country: str='usa', countryofcitizenship: str='united states', name: str='elon', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can get you full list of Forbes billionaires by year.If you want you can search by name or filtered by country or country of citizenship. This api can get list of people page by page and you can search max 100 limit per one request."
    size: You can search max 100 person list per one request.
        
    """
    url = f"https://forbes-worlds-billionaires-list.p.rapidapi.com/billionaires/{year}"
    querystring = {'page': page, 'size': size, }
    if country:
        querystring['country'] = country
    if countryofcitizenship:
        querystring['countryOfCitizenship'] = countryofcitizenship
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forbes-worlds-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

