import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qu_n_huy_n(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lấy quận huyện của 1 tỉnh thành việt nam"
    
    """
    url = f"https://api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com/districts/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x_ph_ng(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lấy Xã Phường  của 1 Quận huyện việt nam"
    
    """
    url = f"https://api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com/wards/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def t_nh_th_nh(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lấy tất cả tỉnh thành việt nam"
    
    """
    url = f"https://api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com/provinces"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-tinh-thanh-quan-huyen-xa-phuong-viet-nam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

