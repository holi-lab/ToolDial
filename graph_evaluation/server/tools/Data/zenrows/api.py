import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scraping(url: str, apikey: str, proxy_country: str='us', autoparse: bool=None, js_render: bool=None, css_extractor: str='%7B%22images%22%3A%22img%20attr%3Asrc%22%2C%22links%22%3A%22a%20attr%3Ahref%22%7D', premium_proxy: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get HTML from any website without blocks"
    
    """
    url = f"https://zenrows1.p.rapidapi.com/"
    querystring = {'url': url, 'apikey': apikey, }
    if proxy_country:
        querystring['proxy_country'] = proxy_country
    if autoparse:
        querystring['autoparse'] = autoparse
    if js_render:
        querystring['js_render'] = js_render
    if css_extractor:
        querystring['css_extractor'] = css_extractor
    if premium_proxy:
        querystring['premium_proxy'] = premium_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zenrows1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

