import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_3000_years(image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the 3000 Years Meme.
		**ALL DOCUMENTATION HERE**: [https://weebyapi.ntmnathan.com/api/docs](https://weebyapi.ntmnathan.com/api/docs)"
    
    """
    url = f"https://weeby.p.rapidapi.com/"
    querystring = {'image': image, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weeby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

