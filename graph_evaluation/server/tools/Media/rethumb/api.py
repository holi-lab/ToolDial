import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def width(pixels: int, image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resize the image based on the specified width."
    pixels: Width of the thumbnail in pixels.
        image: URL of the image to process.
        
    """
    url = f"https://rethumb.p.rapidapi.com/width/{pixels}/{image}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rethumb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def square(pixels: str, image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a square image based on the specified size."
    pixels: Size of the thumbnail square.
        image: URL of the image to process.
        
    """
    url = f"https://rethumb.p.rapidapi.com/square/{pixels}/{image}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rethumb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def height(pixels: str, image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resize the image based on the specified height."
    pixels: Height of the thumbnail in pixels.
        image: URL of the image to process.
        
    """
    url = f"https://rethumb.p.rapidapi.com/height/{pixels}/{image}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rethumb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

