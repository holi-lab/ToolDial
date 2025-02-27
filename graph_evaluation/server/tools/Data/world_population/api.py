import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcountriesbycontinent(continent: str, rank: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain countries' data according to the continent."
    rank: Order the countries according to the rank. Can take two values:

- 1 = ascending order

-  -1 = descending order
        
    """
    url = f"https://world-population3.p.rapidapi.com/continents/{continent}"
    querystring = {}
    if rank:
        querystring['rank'] = rank
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpopulationbycountrycode(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the population's data of a country by its country code (ISO 3166-1 alpha-3 format). More information at https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3."
    
    """
    url = f"https://world-population3.p.rapidapi.com/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

