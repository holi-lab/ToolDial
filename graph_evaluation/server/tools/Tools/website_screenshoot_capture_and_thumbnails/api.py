import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def screenshoot(url: str, viewport: str='1440x900', file_name: str='myscreenshoot', full_page: str='true', inline: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Screenshot full page or thumbnail"
    url: The full URL (including the HTTP & HTTPS Protocol) of the website you want to take a screenshot of. https://gizipp.com
        viewport: Specify your preferred viewport dimensions in pixels. default 1440x900
        file_name: Specify an image name of up to 180 characters.
        full_page: Set true if you want to take a full-page screenshot.

        inline: set to show to display screenshot, download to download screenshot & json to get JSON response with image URL
        
    """
    url = f"https://website-screenshoot-capture-and-thumbnails.p.rapidapi.com/"
    querystring = {'url': url, }
    if viewport:
        querystring['viewport'] = viewport
    if file_name:
        querystring['file_name'] = file_name
    if full_page:
        querystring['full_page'] = full_page
    if inline:
        querystring['inline'] = inline
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-screenshoot-capture-and-thumbnails.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

