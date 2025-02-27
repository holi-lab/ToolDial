import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_php(api: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Blazing fast geocoding open service with world wide coverage and 24x7 Support.Quickly start deploying your apps with our affordable and accurate geocoding api.Support for all languages java, python, php, c, c++, React, Angular etc.Sign up without any credit card and get 2500 free api requests daily."
    api: get your api key for free for 2500 api requests at [https://geokeo.com](https://geokeo.com)
        
    """
    url = f"https://geokeo-forward-geocoding.p.rapidapi.com/search.php"
    querystring = {'api': api, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geokeo-forward-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_php(api: str, lng: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Blazing fast geocoding open service with world wide coverage and 24x7 Support.Quickly start deploying your apps with our affordable and accurate geocoding api.Support for all languages java, python, php, c, c++, React, Angular etc.Sign up without any credit card and get 2500 free api requests daily."
    api: get your api key for free for 2500 api requests at [https://geokeo.com](https://geokeo.com)
        
    """
    url = f"https://geokeo-forward-geocoding.p.rapidapi.com/reverse.php"
    querystring = {'api': api, 'lng': lng, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geokeo-forward-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

