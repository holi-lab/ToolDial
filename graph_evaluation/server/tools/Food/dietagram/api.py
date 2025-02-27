import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def food_api(name: str, lang: str='pl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find food info by name"
    lang: Lang is necessary for search in the certain lang food base.  It will be detected automatically or you could set manually from this set of values (\\\"en\\\", \\\"ru\\\", \\\"pl\\\", \\\"bg\\\", \\\"de\\\", \\\"es\\\", \\\"ua\\\", \\\"by\\\")
        
    """
    url = f"https://dietagram.p.rapidapi.com/apiFood.php"
    querystring = {'name': name, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dietagram.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upc_api(name: str='4019300005154', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find food info by UPC (barcode)"
    
    """
    url = f"https://dietagram.p.rapidapi.com/apiBarcode.php"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dietagram.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

