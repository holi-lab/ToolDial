import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def infobox(wikiurl: str, withname: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a json object representation of the provided wikipedia entry"
    
    """
    url = f"https://wikipedia-infobox.p.rapidapi.com/infobox"
    querystring = {'wikiurl': wikiurl, }
    if withname:
        querystring['withname'] = withname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wikipedia-infobox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

