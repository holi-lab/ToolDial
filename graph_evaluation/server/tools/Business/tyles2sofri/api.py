import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tyles2sofri(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://www.sofri.org/ Tỷ lệ s2 sofri là website chia sẻ dữ liệu bóng đá 7m miễn phí. Bạn có thể truy cập sofri.org 
		để xem tỷ lệ trực tuyến s2 7m, tỷ lệ 2in1, lịch thi đấu, livescore, kết quả bóng đá 24/7.
		Thông tin Liên Hệ:        
		CÔNG TY : s2 sofri
		Địa Chỉ : Phú Kim, Thạch Thất, Hà Nội, Việt Nam
		Điện Thoại: 0972056454
		Website: https://www.sofri.org/
		Email: tylea2sofri@gmail.com
		google Map: https://goo.gl/maps/1BVBcfHSwMJMcuj4A"
    
    """
    url = f"https://tyles2sofri.p.rapidapi.com/tyles2sofri"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tyles2sofri.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

