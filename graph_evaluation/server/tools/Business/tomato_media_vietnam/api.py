import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_292_t_y_s_n(tomato: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Địa điểm làm việc tại 292 Tây Sơn, Đống Đa, Hà Nội"
    
    """
    url = f"https://tomato-media-vietnam.p.rapidapi.com/"
    querystring = {'Tomato': tomato, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomato-media-vietnam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

