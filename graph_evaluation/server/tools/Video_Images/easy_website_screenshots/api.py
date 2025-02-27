import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(url: str, mobile: bool=None, width: int=1280, height: int=-1, returntype: str='base64', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint creates a screenshot of a website and returns the image as Base64 or PNG."
    width: Maximum width is 2048. If omitted, default width of 1280 will be used.
        height: If you specify -1, full page will be captured
        returntype: Return type can be either **base64** or **png**. If omitted, **base64** is default.
        
    """
    url = f"https://easy-website-screenshots.p.rapidapi.com/"
    querystring = {'url': url, }
    if mobile:
        querystring['mobile'] = mobile
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if returntype:
        querystring['returnType'] = returntype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-website-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

