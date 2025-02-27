import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def colors(image_url: str, remove_bg: str=None, show_mask: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract colors from an image"
    image_url: Image URL (required)
        remove_bg: true if background should be removed (default=true)
        show_mask: true if you want to include the mask of detected objects in the response (default=false)
        
    """
    url = f"https://color-extractor-for-apparel-2.p.rapidapi.com/colors"
    querystring = {'image_url': image_url, }
    if remove_bg:
        querystring['remove_bg'] = remove_bg
    if show_mask:
        querystring['show_mask'] = show_mask
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "color-extractor-for-apparel-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

