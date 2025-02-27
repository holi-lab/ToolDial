import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def webiste_categorization_v1(domainname: str, hardrefresh: str=None, outputformat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Define website category at once (v1)"
    domainname: Website's domain name
        hardrefresh: 0 is used for getting the cached category if there is one. 1 is used for demanding the website categorization from scratch. Default: 0
        outputformat: Response output format. Acceptable values: XML or JSON. Defaults to JSON.
        
    """
    url = f"https://website-categorization.p.rapidapi.com/api/v1"
    querystring = {'domainName': domainname, }
    if hardrefresh:
        querystring['hardRefresh'] = hardrefresh
    if outputformat:
        querystring['outputFormat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-categorization.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

