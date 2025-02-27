import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find(url: str, uid: str, client_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Process an image"
    url: A URL to an image
        uid: A unique identifier of your choice
        client_id: *Please contact help@dittolabs.io to get a unique API key for Ditto Photo Reader*
        
    """
    url = f"https://dittolabs-photo-reader-v1.p.rapidapi.com/find"
    querystring = {'url': url, 'uid': uid, 'client_id': client_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dittolabs-photo-reader-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

