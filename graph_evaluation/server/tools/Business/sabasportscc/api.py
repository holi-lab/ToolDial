import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def saba(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://www.sabasports.cc/  Saba Sport CC là nhà cung cấp tỉ lệ kèo nhà cái trực tuyến có đông người chơi nhất hiện nay. Tham gia cá cược bóng đá online tại sảnh Saba Sports Việt Nam - thể thao Saba tại sabasports.cc"
    
    """
    url = f"https://sabasportscc.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sabasportscc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

