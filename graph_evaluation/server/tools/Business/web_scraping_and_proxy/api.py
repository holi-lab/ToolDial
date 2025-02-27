import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def target_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To make a request, you simply include the target URL  API will return the data from that site. Optional configuration options, like proxies, location of the request, and more are possible to include."
    url: The URL to extract the data from. Note that this parameter should include the full HTTP Protocol (http:// or https://). If your URL has parameters, you should encode it. For example the & character would be encoded to %26. Your should check your language for native support of URL encoding or use a third party library.
        
    """
    url = f"https://web-scraping-and-proxy.p.rapidapi.com/v1"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-scraping-and-proxy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

