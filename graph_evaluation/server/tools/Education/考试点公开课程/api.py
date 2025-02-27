import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def 查询专业课(univname: str, majorname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "查询专业课"
    
    """
    url = f"https://Kao-Shi-Dian-Gong-Kai-Ke-Cheng.p.rapidapi.com/query_major"
    querystring = {'univName': univname, 'majorName': majorname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "Kao-Shi-Dian-Gong-Kai-Ke-Cheng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

