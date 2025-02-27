import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_avatar(randomizer: str='anything', name: str='John', bg_color: str='FF2029', text_color: str='000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the desired avatar and returns the image to the client."
    randomizer: The default browser behavior is to cache responses that return from the same endpoint.
If your images are all returning the same, pass this param with ANY value to ensure you bypass this default behaviour and get random images for every request.
        name: Specify the name for the avatar. Only the first character of this parameter will be shown in the avatar.
        bg_color: Specify a HEX CODE color for the background of the avatar. 
Do not include the # of the hex code as it will be ignored on api request.
        text_color: Specify a HEX CODE color for the letter on the avatar. 
Do not include the # of the hex code as it will be ignored on api request.
        
    """
    url = f"https://avatar-io-modern-avatar-placeholder-api.p.rapidapi.com/avatar"
    querystring = {}
    if randomizer:
        querystring['randomizer'] = randomizer
    if name:
        querystring['name'] = name
    if bg_color:
        querystring['bg-color'] = bg_color
    if text_color:
        querystring['text-color'] = text_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "avatar-io-modern-avatar-placeholder-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

