import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def conversion_getlistingfectchallpreviouswithpagination(content_type: str, accept: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://image-to-text-api-from-imagetotext-io.p.rapidapi.com/conversion/getListing"
    querystring = {'Content-Type': content_type, 'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-to-text-api-from-imagetotext-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserinfofectchinfromationaboutcurrentuser(accept: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://image-to-text-api-from-imagetotext-io.p.rapidapi.com/getUserInfo"
    querystring = {'Accept': accept, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-to-text-api-from-imagetotext-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

