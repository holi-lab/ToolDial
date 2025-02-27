import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def snap(width: int=1920, delay: int=0, format: str='png', height: int=1080, hide_scroll_bar: bool=None, url: str='https://www.example.com', exclude_base64: bool=None, wait: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply input url of the webpage that you wish to screenshot.
		API returns a temporary link (valid for 1 hour) to download the screenshot.
		Alternatively, you can also choose to have the image returned in base64 encoding."
    width: Width of the screenshot
        delay: Number of seconds to wait before loading the webpage.
        format: Return image in either png or jpeg format.
        height: Height of the screenshot. Larger height value can capture webpages with longer content.
Up to value of 10000.
        hide_scroll_bar: Choose True to hide browser scroll bar.
        url: Webpage to screenshot.
        exclude_base64: If False, API will return image in base64 encoding. 
        wait: Number of seconds to wait for the webpage to load before taking a screenshot.
Set this to larger number if the webpage is content heavy or you are taking a webpage with large height.
        
    """
    url = f"https://snappy-web-capture.p.rapidapi.com/v1/snap"
    querystring = {}
    if width:
        querystring['width'] = width
    if delay:
        querystring['delay'] = delay
    if format:
        querystring['format'] = format
    if height:
        querystring['height'] = height
    if hide_scroll_bar:
        querystring['hide_scroll_bar'] = hide_scroll_bar
    if url:
        querystring['url'] = url
    if exclude_base64:
        querystring['exclude_base64'] = exclude_base64
    if wait:
        querystring['wait'] = wait
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snappy-web-capture.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

