import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getscores(address: str='1600 Pennsylvania Ave. NW Washington, DC 20500', limit: int=None, offset: int=None, zipcode: str=None, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for scores by address or zip code."
    
    """
    url = f"https://opennwi.p.rapidapi.com/scores/"
    querystring = {}
    if address:
        querystring['address'] = address
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if zipcode:
        querystring['zipcode'] = zipcode
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opennwi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscoredetails(is_id: int, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search score details by rankID.
		**Key**:
		- d2a: The mix of employment types and occupied housing.
		- d2b: The mix of employment types in a block group
		- d3b: Street intersection density
		- d4a: workers who carpool
		- bikeShareRank: Ranking (out of 20) for quality of public bikeshare services in the region"
    
    """
    url = f"https://opennwi.p.rapidapi.com/details/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opennwi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

