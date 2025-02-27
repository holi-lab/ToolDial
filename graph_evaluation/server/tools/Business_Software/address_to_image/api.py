import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getimage(output: str='image', type: str='jpeg', marker: int=0, zoom: int=1, address: str='Times Square', width: int=1024, height: int=1024, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates an image for a given address. Images are output in JPEG as a base64 encoded string."
    type: Image type, either **png **or **jpeg**
        marker: Whether or not to display a marker on the image at the desired address, 0 or 1
        zoom: The level of image zoom. Default is 1, range is 0-5
        address: Address or location.
        width: Image width - maximum 1024px
        height: Image height- maximum 1024px
        
    """
    url = f"https://address-to-image.p.rapidapi.com/"
    querystring = {}
    if output:
        querystring['output'] = output
    if type:
        querystring['type'] = type
    if marker:
        querystring['marker'] = marker
    if zoom:
        querystring['zoom'] = zoom
    if address:
        querystring['address'] = address
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-to-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

