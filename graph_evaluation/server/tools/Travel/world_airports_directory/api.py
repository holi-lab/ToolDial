import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchairports(matchingkeywordairportorcityorcode: str, sortby: str, page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Primary API to  get all the matching airports based on the keyword you pass in the params. API runs best matching algorithm to find most relevant airports matching with either city, city code or airport code and so on.
		**BONUS**: It returns complete `Airport Name, Lat, Long, Timezone, Country, Country Code, Airport Code and City Code.`
		
		NOTE: This API is case insensitive.
		
		For e.g.
		1. When user trying with city `/v1/airports/dubai`
		2. When user trying with Airport Code `/v1/airports/dxb`
		3. When user trying with city code `/v1/airports/omdb`"
    sortby: Valid format to sort is `field:order`
e.g. `AirportName:asc`, `AirportName:desc`
where `asc` for sorting in ascending order
`desc` for sorting in descending order
        page: This controls the pagination of results.
Default is `1`
        limit: This indicates how many results you would like to query in one time.
Default value is `10` if limit is not provided.
Maximum limit is `20`
        
    """
    url = f"https://world-airports-directory.p.rapidapi.com/v1/airports/{matchingkeywordairportorcityorcode}"
    querystring = {'sortBy': sortby, 'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-airports-directory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallairports(limit: int, page: int, sortby: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to explore all airports we have in our huge directory."
    sortby: Valid format to sort is `field:order`
e.g. `name:asc`, `name:desc`
where `asc` for sorting in ascending order
`desc` for sorting in descending order
        
    """
    url = f"https://world-airports-directory.p.rapidapi.com/v1/airports"
    querystring = {'limit': limit, 'page': page, 'sortBy': sortby, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-airports-directory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

