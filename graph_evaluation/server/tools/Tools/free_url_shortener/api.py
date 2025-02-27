import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def http_ulvis_net_api_write_get(url: str, custom: str=None, password: bool=None, private: int=None, type: str='type', via: str='via', uses: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shorten url  url= url-wan't shorten &custom= your custom name& private= if wan't be public them 0 if private them 1http://ulvis.net/api/write/post"
    url: url 
        url: The url you want to shrink.
        custom: custum
        password: set url to password
        private: 0 be url public but if 1  be them private
        type: esponse type (json|xml). optional, default: json.
        custom: Custom name
        private: Set url to private (not listed) (1 or 0)
        via: Adds a signature to track your application
        password: Set url password
        uses: Number of uses for the url
        
    """
    url = f"https://free-url-shortener.p.rapidapi.com/API/write/get"
    querystring = {'url': url, }
    if custom:
        querystring['custom'] = custom
    if password:
        querystring['password'] = password
    if private:
        querystring['private'] = private
    if type:
        querystring['type'] = type
    if custom:
        querystring['custom'] = custom
    if private:
        querystring['private'] = private
    if via:
        querystring['via'] = via
    if password:
        querystring['password'] = password
    if uses:
        querystring['uses'] = uses
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

