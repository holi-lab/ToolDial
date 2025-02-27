import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_prediction(format: str='json', locale: str='en-US', theme: str='common', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can pass theme, locale and output format. For example "common" for theme returns abstract prediction. Use locale parameter for choose preferred language. Output format by default JSON, pass "text" to get result as plaint text."
    
    """
    url = f"https://all-predictions-in-one-place.p.rapidapi.com/get-prediction/{theme}/{locale}/{format}"
    querystring = {}
    if format:
        querystring['format'] = format
    if locale:
        querystring['locale'] = locale
    if theme:
        querystring['theme'] = theme
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-predictions-in-one-place.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_available_themes(locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of themes available to predictions. Pass locale to return list for preferred language."
    
    """
    url = f"https://all-predictions-in-one-place.p.rapidapi.com/get-available-themes/{locale}"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-predictions-in-one-place.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_available_locales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available locales for predictions"
    
    """
    url = f"https://all-predictions-in-one-place.p.rapidapi.com/get-available-locales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-predictions-in-one-place.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

