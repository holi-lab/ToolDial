import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_placeholder_text(length: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates placeholder text based on "length" query param"
    length: Specifies what size placeholder text you want in the response.
The \"length\" param must be one of the following:
- small
- medium
- large
        
    """
    url = f"https://placeholder-text-api-for-your-application.p.rapidapi.com/placeholder"
    querystring = {'length': length, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "placeholder-text-api-for-your-application.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

