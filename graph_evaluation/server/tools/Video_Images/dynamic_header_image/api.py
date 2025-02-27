import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geo_location_hello_banner(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates a SVG banner with beautiful gradiant with 'Hello' written in local language of the user hitting the API.
		It checks the IP address location and present user with Hello screen.
		Default language is English, in case there is any error determining user location."
    
    """
    url = f"https://dynamic-header-image.p.rapidapi.com/?locHello=true"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dynamic-header-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_text_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This generates custom SVG image of the text you provided with a beautiful gradient background that changes on every request you send."
    
    """
    url = f"https://dynamic-header-image.p.rapidapi.com/?text=Hello%20RapidAPI%20User"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dynamic-header-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

