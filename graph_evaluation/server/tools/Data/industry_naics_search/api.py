import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def filter_and_query_naics_description(naics: int=721110, q: str='Hotel', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will allow you to search by freeform text, eg. Hotel and get all the list of NAICS code, hash, and description back.
		You can also filter the search if you have an NAICS code and want to get the description and the hash of the industry."
    naics: This parameter take full 6 digit NAICS code only.
Optional if q parameter is present.
        q: When freeform search without NAICS param, minimum 3 character to search.
Optional if NAICS param is present.
        
    """
    url = f"https://industry-naics-search.p.rapidapi.com/industry/v1/naics"
    querystring = {}
    if naics:
        querystring['naics'] = naics
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "industry-naics-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

