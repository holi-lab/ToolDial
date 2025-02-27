import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def viewrecords(scope: str, authtoken: str, applinkname: str, zc_ownername: str, viewlinkname: str, raw: str='True', criteria: str='(Country == "US")', startindex: str='0', limit: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replace https://zohocreator.p.rapidapi.com/ with https://creator.zoho.com/api/ if you're using native system
		
		Get Records from a Zoho View/Report"
    
    """
    url = f"https://zohocreator.p.rapidapi.com/json/{applinkname}/view/{viewlinkname}"
    querystring = {'scope': scope, 'authtoken': authtoken, 'zc_ownername': zc_ownername, }
    if raw:
        querystring['raw'] = raw
    if criteria:
        querystring['criteria'] = criteria
    if startindex:
        querystring['startindex'] = startindex
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zohocreator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

