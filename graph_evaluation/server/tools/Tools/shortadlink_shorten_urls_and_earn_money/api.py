import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shorten_url(api: str, url: str, format: str=None, type: str=None, alias: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Primary method for shortening URL with custom alias."
    api: You can get your API key by registering on [ShortAdLink](https://shortad.link/)
        url: Enter your long URL link that you want to shorten.
        format: Default response is in **JSON** format. No need to type anything.
You can also get response in text format. Just type **text** in format field.
        type: Default advertisement type is banner.
Formats available:

> **Default**: Banner (No need to type anything)
> **0**: No advertisement
> **1**: Interstitial advertisement
        alias: You can set custom name for your shortened URL.
Alias **min.** length: 4
Alias **max** length: 12
        
    """
    url = f"https://shortadlink-shorten-urls-and-earn-money.p.rapidapi.com/api?"
    querystring = {'api': api, 'url': url, }
    if format:
        querystring['format'] = format
    if type:
        querystring['type'] = type
    if alias:
        querystring['alias'] = alias
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shortadlink-shorten-urls-and-earn-money.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

