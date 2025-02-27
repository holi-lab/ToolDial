import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_mlem(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns one most recently published mlem image in JSON"
    
    """
    url = f"https://mlemapi.p.rapidapi.com/newmlem"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlemapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mlem_id(mlemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns one mlem image by ID in JSON"
    mlemid: Mlem ID
        
    """
    url = f"https://mlemapi.p.rapidapi.com/mlemid"
    querystring = {'mlemid': mlemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlemapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all tags in JSON"
    
    """
    url = f"https://mlemapi.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlemapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_mlem(brightness: str=None, maxheight: int=None, minwidth: int=None, minheight: int=None, tag: str=None, maxwidth: int=None, orientation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns one random mlem image in JSON"
    brightness: Image brightness: dark or bright
        maxheight: Maximum height
        minwidth: Maximum width
        minheight: Minimum height
        tag: Tag of mlem
        maxwidth: Minimum width
        orientation: Image orientation: square, landscape, portrait
        
    """
    url = f"https://mlemapi.p.rapidapi.com/randommlem"
    querystring = {}
    if brightness:
        querystring['brightness'] = brightness
    if maxheight:
        querystring['maxheight'] = maxheight
    if minwidth:
        querystring['minwidth'] = minwidth
    if minheight:
        querystring['minheight'] = minheight
    if tag:
        querystring['tag'] = tag
    if maxwidth:
        querystring['maxwidth'] = maxwidth
    if orientation:
        querystring['orientation'] = orientation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlemapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

