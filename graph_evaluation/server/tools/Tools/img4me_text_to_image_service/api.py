import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main(text: str, font: str='trebuchet', size: int=12, fcolor: str='000000', bcolor: str='FFFFFF', type: str='png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API will return the URL of the image generated."
    text: Text to be converted into image.
        font: Font for the image generated. Supported word: arial, comic, dyslexic, georgia, impact, lucida, simsun, tahoma, times, trebuchet, verdana
        size: Font size.
        fcolor: Font color in HEX.
        bcolor: Background color in HEX. Leave empty for transparent background.
        type: Image format: gif, jpg, png
        
    """
    url = f"https://img4me.p.rapidapi.com/"
    querystring = {'text': text, }
    if font:
        querystring['font'] = font
    if size:
        querystring['size'] = size
    if fcolor:
        querystring['fcolor'] = fcolor
    if bcolor:
        querystring['bcolor'] = bcolor
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "img4me.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

