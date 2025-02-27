import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def w388gnet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[W388](https://w388g.net/) - Thương hiệu cá cược làm chao đảo làng cá cược nhờ nhiều trò chơi hấp dẫn cũng như dịch vụ đặc biệt. Liệu có thật sự như những lời đánh giá này?
		Website: https://w388g.net/
		Thông tin liên hệ
		Địa chỉ: 6 Ng.48 P. D. Quảng Hàm, Quan Hoa, Cầu Giấy, Hà Nội
		Email: lythetuong2901@gmail.com
		#w388, #w388_net, #nha_cai_w388"
    
    """
    url = f"https://w388gnet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "w388gnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

