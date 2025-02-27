import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_zip_file_of_the_tweets(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint will give you the bytes of a zip file that have the tweets"
    
    """
    url = f"https://twitter-tweets-base.p.rapidapi.com/attack/{user}"
    querystring = {}
    if user:
        querystring['user'] = user
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-tweets-base.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

