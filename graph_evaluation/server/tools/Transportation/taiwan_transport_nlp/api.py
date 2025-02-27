import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def 透過自然語言查詢時刻表與轉乘資訊(ask: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "from：出發站點名稱
		destination ：目的地站點名稱
		time：出發時間
		type：大眾運輸種類（包含 hsr(高鐵)、tra(台鐵)、mrt(北捷)、l_bus(客運)）
		timetable：時刻表"
    
    """
    url = f"https://taiwan-transport-nlp.p.rapidapi.com/nlp"
    querystring = {'ask': ask, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taiwan-transport-nlp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

