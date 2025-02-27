import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes in the url of video needed to be download and returns the JSON response of it.
		Note: There may be a chance that you need to pass cookies in request header to fulfill the request such as downloading Instagram Private Video, Stories etc.(Some times public video also needs cookies) Also use **cookies** it is case sensitive and must be in plural."
    
    """
    url = f"https://vidsnap.p.rapidapi.com/fetch"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vidsnap.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

