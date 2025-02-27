import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text(input: str, locale: str='en', location: str='50.3,9.0', clientfeatures: str='all', timezone: str='+120', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Plaintext response"
    input: Just about any phrase, query or command in any language
        locale: Language you want to communicate in. 30 languages supported!
        location: Append longitude and latitude to get more location aware responses. Otherwise the IP will used to guess the location.
        clientfeatures: see documentation
        timezone: offset to CET
        
    """
    url = f"https://jeannie.p.rapidapi.com/text/"
    querystring = {'input': input, }
    if locale:
        querystring['locale'] = locale
    if location:
        querystring['location'] = location
    if clientfeatures:
        querystring['clientFeatures'] = clientfeatures
    if timezone:
        querystring['timeZone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jeannie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def json(input: str, timezone: str='+120', locale: str='en', location: str='53.0,9.0', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Http request"
    input: Any free text, question or command
        timezone: Specify the timezone offset of your client compared to UTC in minutes. E.g. +120 for Germany
        locale: Specify language (de, en, fr, en_US, ...) optional: auto-detected if omitted
        location: Append longitude and latitude to get more location aware responses. Otherwise the IP will used to guess the location.
        page: Sometimes you may page through lists of links, images, pages etc
        
    """
    url = f"https://jeannie.p.rapidapi.com/api"
    querystring = {'input': input, }
    if timezone:
        querystring['timeZone'] = timezone
    if locale:
        querystring['locale'] = locale
    if location:
        querystring['location'] = location
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jeannie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

