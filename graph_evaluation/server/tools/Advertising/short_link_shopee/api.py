import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v_d_2_link_shopee_r_t_g_n(link: str, appid: str, secret: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ví dụ 1"
    
    """
    url = f"https://short-link-shopee.p.rapidapi.com/shortlink"
    querystring = {'link': link, 'appid': appid, 'secret': secret, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short-link-shopee.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v_d_1(appid: str, link: str, secret: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ví dụ 1"
    
    """
    url = f"https://short-link-shopee.p.rapidapi.com/shortlink"
    querystring = {'appid': appid, 'link': link, 'secret': secret, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short-link-shopee.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

