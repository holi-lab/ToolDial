import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def businesses(keyword: str=None, name: str=None, city: str=None, state: str=None, postalcode: str=None, sort: str='value_distance', category: str='Category', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    keyword: A term to search against business names and categories.
        name: The name of a business.
        city: The name of the city to use as a search point.
        state: The two-letter abbreviation of the state in USA to use as a search point.
        postalcode: The US postal code of a location.  ( 5 or 9 digit)
        category: The category that defines (or relates to) the commercial offering of a business.
        
    """
    url = f"https://soleo_api-local-search-v1.p.rapidapi.com/businesses"
    querystring = {}
    if keyword:
        querystring['Keyword'] = keyword
    if name:
        querystring['Name'] = name
    if city:
        querystring['City'] = city
    if state:
        querystring['State'] = state
    if postalcode:
        querystring['PostalCode'] = postalcode
    if sort:
        querystring['Sort'] = sort
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soleo_api-local-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

