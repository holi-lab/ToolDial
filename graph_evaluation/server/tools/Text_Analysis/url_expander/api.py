import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_expander(short_url: str, get_title: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide short_url through Query Param.
		If the target URL is no more active (or removed/moved) it will return 
		{ target_url_alive : False }
		URL's response_time is in milliseconds"
    short_url: Provide Shortened Url
        get_title: If get_title set to True, then the title of the document will be extracted from url.
Note: Don't set this flag true if the title is not needed, to avoid slow-down
        
    """
    url = f"https://url-expander1.p.rapidapi.com/url_expander"
    querystring = {'short_url': short_url, }
    if get_title:
        querystring['get_title'] = get_title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-expander1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

