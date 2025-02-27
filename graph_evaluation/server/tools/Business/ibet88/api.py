import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ibet88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IBet88 - Nhà cái uy tín số 1 Việt Nam ⭐️ Chuyên cá cược thể thao, bóng đá, live casino, đá gà trực tuyến –  trực tiếp, slot game & bắn cá. Tham gia ngay
		#đăng ký nạp tiền #rút tiền #tải ứng dụng #ibet88pro #ibet88 /m/033_l8  /m/033_l8 #Casino
		
		Địa chỉ 551 Ngọc Lâm, Long Biên, Hà Nội, Việt Nam
		Hotline 0876497649
		Email: ibet88pro1@gmail.com
		Webiste [https://ibet88.pro/ ](url)
		[https://ibet88.pro/dang-ky-ibet88/ 
		](url)https://ibet88.pro/nap-tien-ibet88/ 
		[https://ibet88.pro/rut-tien-ibet88/ 
		](url)[https://ibet88.pro/tai-ung-dung-ibet88/ 
		](url)"
    
    """
    url = f"https://ibet88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ibet88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

