import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_generation_endpoint(width: int=500, backgroundimage: str='https://source.unsplash.com/500x500/', height: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image endpoint.
		Check docs.bruzu.com for more information."
    
    """
    url = f"https://bruzu.p.rapidapi.com/"
    querystring = {}
    if width:
        querystring['width'] = width
    if backgroundimage:
        querystring['backgroundImage'] = backgroundimage
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bruzu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

