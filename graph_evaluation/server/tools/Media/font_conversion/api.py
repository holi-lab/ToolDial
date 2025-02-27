import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_fonts(public: str='0', start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the fonts in the system or added by you."
    public: Should include public fonts in the list or not?
        start: The result is paged. Start specifies the index to start the listing from.
        
    """
    url = f"https://fontconversion.p.rapidapi.com/font/list.json"
    querystring = {}
    if public:
        querystring['public'] = public
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fontconversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

