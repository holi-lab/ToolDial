import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_avatar_based_on_username(uname: str, g: str, fe: str='gif', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will generate the avatar based on username and will also return the avatar in request image format."
    fe: It's not required and default value for image format is png
        
    """
    url = f"https://funky-pixel-avatars.p.rapidapi.com/api/v1/avatar/generate/user"
    querystring = {'uname': uname, 'g': g, }
    if fe:
        querystring['fe'] = fe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funky-pixel-avatars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_avatar(g: str, fe: str='jpeg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate avatar api get gender and image type from user and generates the avatar of given gender and image type."
    fe: This query parameter can be blank, default image type is png
        
    """
    url = f"https://funky-pixel-avatars.p.rapidapi.com/api/v1/avatar/generate"
    querystring = {'g': g, }
    if fe:
        querystring['fe'] = fe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funky-pixel-avatars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

