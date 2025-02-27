import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hcm66bet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "hcm66 bet tặng tiền cược miễn phí cho thành viên mới khi tham gia đăng ký trải nghiệm game đổi thưởng ăn tiền thật như casino, cá cược thể thao, slot, lô đề. Nhà cái HCM66 là thương hiệu cá cược hợp pháp tại Việt Nam.
		Thông tin liên hệ:
		CÔNG TY: HCM66 BET
		Địa chỉ: 167 P.Trần Đăng Ninh, Phú La, Hà Đông, Hà Nội
		Điện thoại: 0958555555
		website: [https://www.hcm66.bet/](url) 
		email: hcm66bet@gmail.com
		google map: https://goo.gl/maps/JAGtDUjxLPA8P1Zr9"
    
    """
    url = f"https://hcm66bet.p.rapidapi.com/hcm66bet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hcm66bet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

