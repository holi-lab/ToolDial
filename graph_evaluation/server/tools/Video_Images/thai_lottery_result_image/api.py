import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gen_image(date: str='01062564', bgimg: str=None, rmber: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gen Image"
    
    """
    url = f"https://thai-lottery-result-image.p.rapidapi.com/"
    querystring = {}
    if date:
        querystring['date'] = date
    if bgimg:
        querystring['bgimg'] = bgimg
    if rmber:
        querystring['rmber'] = rmber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery-result-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

