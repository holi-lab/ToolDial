import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpageandposition(domainname: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds where a domain name is listed in Bing search results"
    domainname: the domain name you wish to find in the search results
        query: The phrase or keywords you wish to search by
        
    """
    url = f"https://bingsearch1.p.rapidapi.com/bing/v1/"
    querystring = {'DomainName': domainname, 'Query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bingsearch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

