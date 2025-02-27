import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def using_color_param(colour: str='FF9090', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Apply any colour to the SVG"
    
    """
    url = f"https://background-pattern1.p.rapidapi.com/"
    querystring = {}
    if colour:
        querystring['colour'] = colour
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-pattern1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_params(color: str='#FF9090', key: str='RapidAPI', type: str='svg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use all possible path"
    
    """
    url = f"https://background-pattern1.p.rapidapi.com/"
    querystring = {}
    if color:
        querystring['color'] = color
    if key:
        querystring['key'] = key
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-pattern1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def using_key_param(key: str='RapidAPI', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send **KEY** param to create exact SVG.
		That can be useful if you want to have the same SVG twice"
    
    """
    url = f"https://background-pattern1.p.rapidapi.com/"
    querystring = {}
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-pattern1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def using_type_param(type: str='svg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using **Type** param you can choose if you want to fetch the SVG image or the URI"
    
    """
    url = f"https://background-pattern1.p.rapidapi.com/"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "background-pattern1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

