import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_properties(includeddepartments: str, propertytypes: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return properties matching given criteria."
    
    """
    url = f"https://melo.p.rapidapi.com/documents/properties"
    querystring = {'includedDepartments[]': includeddepartments, }
    if propertytypes:
        querystring['propertyTypes[]'] = propertytypes
    if propertytypes:
        querystring['propertyTypes[]'] = propertytypes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

