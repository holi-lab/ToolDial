import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def website_contacts_v1(domainname: str, outputformat: str=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full contact information from page, domain or web (v1)"
    domainname: The website's domain name.
        outputformat: Response output format. Acceptable values: XML or JSON. Defaults to JSON.
        is_from: 0 is used for getting the cached contacts information if there is one, 1 is used for demanding the website contacts information from scratch. Default: 0
        
    """
    url = f"https://website-contacts.p.rapidapi.com/api/v1"
    querystring = {'domainName': domainname, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-contacts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

