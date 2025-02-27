import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def twitter_interest_graph(twitterhandle: str='kimkardashian', token: str='3hbv1ionxeoyl9pzsy49e7bl5yh45i830nxuono4vzq309ii80whj9mu022rwgd3', app_key: str='dee1eb769deb1c7ed850fc2ab18c31e5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compute vuGraph of Twitter handle"
    twitterhandle: Twitter handle for which you want to compute interest graph
        token: Your auth token
        app_key: Your auth key
        
    """
    url = f"https://twitter-vugraph.p.rapidapi.com/vugraph"
    querystring = {}
    if twitterhandle:
        querystring['twitterhandle'] = twitterhandle
    if token:
        querystring['token'] = token
    if app_key:
        querystring['app_key'] = app_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-vugraph.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

