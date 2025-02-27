import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def badge_from_shields_io(is_id: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a particular attribute from the playstore data in a format that is recognized by the shields.io badges. This can be used in combination with shields.io for generating android app related badges with realtime values for your app's README file."
    id: The package name of your android application.
        type: can be one among : **[ downloads, package, version, size, lastUpdated, rating, noOfUsersRated, developer ]**
        
    """
    url = f"https://app-details-from-playstore.p.rapidapi.com/{type}"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-details-from-playstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def json_with_all_details(is_id: str, forceupdate: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all the available details."
    id: This is where you will need to pass your app's unique id (or) sometimes called as package-name.
        forceupdate: Usually, this API caches the app's data for about an hour. Hence, if you want to override this cache (or) want **non-cached fresh** data of your app, you will have to set this parameter to ***true***.
        
    """
    url = f"https://app-details-from-playstore.p.rapidapi.com/json"
    querystring = {'id': is_id, }
    if forceupdate:
        querystring['forceUpdate'] = forceupdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-details-from-playstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

