import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_image_with_url(image_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET Passport MRZ by sending document image as a image_url"
    image_url: **Use our Text Analyze API to detect & extract MRZ from passport pictures .**

*Image must be a regular JPEG or PNG image. Usually such images have extensions: .jpg, .jpeg, .png. The service checks input file by MIME type …
        
    """
    url = f"https://passport-mrz-extract.p.rapidapi.com/"
    querystring = {'image_url': image_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "passport-mrz-extract.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

