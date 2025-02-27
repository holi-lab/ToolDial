import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generatememe(top: str, bottom: str, meme: str, font_size: str='50', font: str='Impact', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The meme is generated from a set of predefined images or an image uploaded via the `uploadImage` API. To get a list of all predefined and uploaded images available to create the meme, please execute the `getImages` API or go to apimeme.com to see which images are available."
    top: The text that will be rendered at the top of the image
        bottom: The text that will be rendered at the bottom of the image
        meme: The image that will be used to generate the meme
        font_size: The size of the font for the top and the bottom texts
        font: The font that will be used for the top and bottom texts
        
    """
    url = f"https://ronreiter-meme-generator.p.rapidapi.com/meme"
    querystring = {'top': top, 'bottom': bottom, 'meme': meme, }
    if font_size:
        querystring['font_size'] = font_size
    if font:
        querystring['font'] = font
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ronreiter-meme-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfonts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available fonts."
    
    """
    url = f"https://ronreiter-meme-generator.p.rapidapi.com/fonts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ronreiter-meme-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available meme images."
    
    """
    url = f"https://ronreiter-meme-generator.p.rapidapi.com/images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ronreiter-meme-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

