import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_pdf(width: int=1024, height: int=780, url: str='https://google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate PDF by providing URL of any website."
    
    """
    url = f"https://web-capture2.p.rapidapi.com/pdf"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-capture2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def take_image_screenshot(width: int=1024, url: str='https://google.com', height: int=780, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Screenshot of any website by providing URL and the  image ratio (width, height)"
    
    """
    url = f"https://web-capture2.p.rapidapi.com/image"
    querystring = {}
    if width:
        querystring['width'] = width
    if url:
        querystring['url'] = url
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-capture2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

