import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image2text(imageurl: str, filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the image url and filename and get back the text in the image as html. We do not store anything on our server."
    imageurl: The url must be accessible  online. Cannot be behind a firewall or authentication.

        filename: the filename must be a string plus a dot plus a file extension. Example: image1.png or myimage.bmp


        
    """
    url = f"https://ocrly-image-to-text.p.rapidapi.com/"
    querystring = {'imageurl': imageurl, 'filename': filename, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocrly-image-to-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

