import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def posts(postid: int=1, ioiii: str=None, enum: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "posts"
    postid: test
        enum: iuuu
        
    """
    url = f"https://post-generator.p.rapidapi.com/posts"
    querystring = {}
    if postid:
        querystring['postID'] = postid
    if ioiii:
        querystring['ioiii'] = ioiii
    if enum:
        querystring['enum'] = enum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "post-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

