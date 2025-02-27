import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def capture(url: str, start: int, end: int, duration: int=None, size: str=None, crop: str=None, fps: int=15, trailer: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Main API endpoint used to generate GIFs"
    url: The video URL you would like to generate a GIF from
        start: The starting time (in seconds) of your GIF
        end: The ending time (in seconds) of your GIF
        duration: The duration (in seconds) of your GIF
        size: resize the GIF to the dimensions of your choice (accepts width and height in pixels) (Default: 300x200)
        crop: Crop the GIF to the size of your choice (accepts width and height in pixels)
        fps: specify a GIF quality (Frames per Second) of your choice (Default: 15)
        trailer: Set to "1" to create default trailer, or specify custom trailer parameters
        
    """
    url = f"https://apilayer-giflayer-v1.p.rapidapi.com/capture"
    querystring = {'url': url, 'start': start, 'end': end, }
    if duration:
        querystring['duration'] = duration
    if size:
        querystring['size'] = size
    if crop:
        querystring['crop'] = crop
    if fps:
        querystring['fps'] = fps
    if trailer:
        querystring['trailer'] = trailer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-giflayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

