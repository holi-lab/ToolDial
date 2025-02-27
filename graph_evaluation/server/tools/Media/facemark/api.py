import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def process_image_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON object that contains detected facial features for faces on the image specified by URL  (visual demo: http://apicloud.me/apis/facerect/demo/)"
    url: URL of the image you would like to find facial features on. For supported image files and limitations please refer to the documentation: http://apicloud.me/apis/facemark/docs
        
    """
    url = f"https://apicloud-facemark.p.rapidapi.com/process-url.json"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apicloud-facemark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

