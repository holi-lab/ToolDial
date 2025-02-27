import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def l_ng_qu_c_tr(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tôi là Lăng Quốc Trụ, sáng lập website nhacaiuytinhot.com - nơi đánh giá và tổng hợp danh sách các nhà cái uy tín nhất Việt Nam hiện nay. Với 7 năm kinh nghiệm cá cược hi vọng sẽ đem đến cho các bạn những thông tin hữu ích.
		Website: https://nhacaiuytinhot.com 
		Địa chỉ: Hà Nội
		Hotline: 088 854 0267
		https://www.facebook.com/langquoctru 
		https://www.pinterest.com/langquoctru/ 
		https://social.msdn.microsoft.com/profile/langquoctru/ 
		https://diigo.com/user/langquoctru 
		https://social.technet.microsoft.com/profile/langquoctru/ 
		https://vi.gravatar.com/langquoctru 
		https://www.reddit.com/user/langquoctru 
		https://soundcloud.com/langquoctru/ 
		https://www.behance.net/langquoctru 
		https://medium.com/@langquoctru/ 
		https://www.openstreetmap.org/user/langquoctru 
		https://www.twitch.tv/langquoctru/about 
		https://issuu.com/langquoctru"
    
    """
    url = f"https://langquoctru.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "langquoctru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

