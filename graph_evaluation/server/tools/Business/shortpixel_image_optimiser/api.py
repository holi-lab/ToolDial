import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reducer(url: str, lossy: int=0, wait: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main API, and allows you to shrink an image based on the URL of the image. The images has to be available online in order to be shrunk via this API."
    url: The URL where the original image is located. This needs to be a valid URL and urlencoded
        lossy: Controls whether the image compression will use a lossy or lossless algorithm
        wait: Set it to 1 to wait for the conversion to be done before the API returns, 0 to return immediately. If set to 1, the API call will wait up to 10 seconds for the image to be converted
        
    """
    url = f"https://pagepeeker-shortpixel-image-optimiser-v1.p.rapidapi.com/v1/reducer.php"
    querystring = {'url': url, }
    if lossy:
        querystring['lossy'] = lossy
    if wait:
        querystring['wait'] = wait
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepeeker-shortpixel-image-optimiser-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

