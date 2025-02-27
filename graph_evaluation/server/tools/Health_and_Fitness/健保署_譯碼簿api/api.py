import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def 取出譯碼簿版本和目錄(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "取出譯碼簿版本和目錄"
    
    """
    url = f"https://Jian-Bao-Shu-Yi-Ma-Bo-api.p.rapidapi.com/outline"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "Jian-Bao-Shu-Yi-Ma-Bo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def 查詢譯碼簿資料表(tablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "查詢譯碼簿資料表"
    tablename: 可查詢的譯碼簿資料表名稱如下：

> HOSP_ DATA_TYPE

> APPL_TYPE

> APPL_MODE

> CASE_TYPE

> CURE_ITEM

> FUNC_TYPE

        
    """
    url = f"https://Jian-Bao-Shu-Yi-Ma-Bo-api.p.rapidapi.com/table/{tablename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "Jian-Bao-Shu-Yi-Ma-Bo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

