import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_cosmetic_ingredients_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all Cosmetic  ingredients Name
		화장품 성분명칭 전체 데이터를 불러옵니다."
    
    """
    url = f"https://cosmetic-ingredients2.p.rapidapi.com/cosmetic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cosmetic-ingredients2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cosmetic_ingredients_id_value(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cosmetic-ingredients name id value 
		화장품 성분데이터의 명칭을 id 로 확인하여 불러옵니다."
    
    """
    url = f"https://cosmetic-ingredients2.p.rapidapi.com/cosmetic/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cosmetic-ingredients2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

