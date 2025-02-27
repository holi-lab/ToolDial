import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def import_url(import_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in the url to be imported and meta tags extracted, e.g: https://bing.com/ Must contain the http or https and be a valid url or will return null. This data will be cached unless force is passed in."
    
    """
    url = f"https://seo-meta-tag-search.p.rapidapi.com/"
    querystring = {'import_url': import_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-meta-tag-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

