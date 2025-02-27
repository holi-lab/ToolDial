import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchbycompanyname(q: str, resource_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint can be used to search for an Israeli company name or company identifier. 
		
		The resource ID should always be set to f004176c-b85f-4542-8901-7b3176f9a054  as the default value."
    
    """
    url = f"https://israel-company-data.p.rapidapi.com/api/3/action/datastore_search"
    querystring = {'q': q, 'resource_id': resource_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "israel-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

