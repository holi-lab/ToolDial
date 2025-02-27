import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def multiple_guids(number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to retrieve a multiple GUIDs, returned as a a string array.
		Required query string parameter: *number* - needs to be between 0 and 1000.
		GET only query.
		Return content-type is text/json."
    
    """
    url = f"https://guid-generator-tool.p.rapidapi.com/httptriggermultiguidgenerate"
    querystring = {'number': number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "guid-generator-tool.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_guid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to retrieve a single GUID, returned as a simple string.
		No parameters required.
		GET only query.
		Return content-type is text/plain."
    
    """
    url = f"https://guid-generator-tool.p.rapidapi.com/httptriggerguidgenerate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "guid-generator-tool.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

