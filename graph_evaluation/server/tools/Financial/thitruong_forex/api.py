import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def thitruong_forex(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://thitruongforex.info/ nơi chia sẻ kiến thức giao dịch ngoại hối (forex) và hướng dẫn các sàn uy tín như: Exness, FBS,... hướng dẫn mở tài khoản cho người mới bắt đầu.
		Địa chỉ: Hà Nội, Việt Nam
		Website: https://thitruongforex.info/
		Email: hotro.thitruongforex@gmail.com
		#Thitruongforex.info #Thitruongforex"
    
    """
    url = f"https://thitruong-forex.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thitruong-forex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

