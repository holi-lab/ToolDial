import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of included proxy servers by location"
    
    """
    url = f"https://url-website-domain-browser-screenshot.p.rapidapi.com/proxies.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-website-domain-browser-screenshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def take_screenshot(website: str, resolution: str='1680×1050', proxy: str=None, fullpage: bool=None, iptype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specify url with additional options to take a fully customizable screenshot of any website domain or url"
    
    """
    url = f"https://url-website-domain-browser-screenshot.p.rapidapi.com/screenshot.php"
    querystring = {'website': website, }
    if resolution:
        querystring['resolution'] = resolution
    if proxy:
        querystring['proxy'] = proxy
    if fullpage:
        querystring['fullpage'] = fullpage
    if iptype:
        querystring['iptype'] = iptype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-website-domain-browser-screenshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

