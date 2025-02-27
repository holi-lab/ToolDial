import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mikiishop(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mikiishop với kinh nghiệm 3 năm order các mặt hàng túi xách, ví, giày nam nữ cao cấp kể cả hàng khó, tự tin sẽ giúp các quý anh chị lựa chọn được những sản phẩm fake cao cấp nhất
		#MikiiShop #Mikii Shop  #GUCCI #DIOR #LOUIS VUITTON #HERMES #CHANEL #MCM #YSL #FENDI #Túi_Xách_Fendi
		
		Địa chỉ : 315 Xóm Chiếu, Phường 16, Quận 4, Thành phố Hồ Chí Minh 700000
		Hotline 0382210283
		Gmail: mikiishop.com@gmail.com
		GG map https://www.google.com/maps?cid=11710343567588533487 
		Website https://mikiishop.com/ 
		https://mikiishop.com/louis-vuitton/ 
		https://mikiishop.com/dior/ 
		https://mikiishop.com/chanel/ 
		https://www.youtube.com/channel/UCvNpIZBSNsKjOK8pV560iIg/about
		https://www.instagram.com/mikiishopcom/
		https://twitter.com/mikiishop
		https://www.pinterest.com/mikiishop/"
    
    """
    url = f"https://mikiishop.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mikiishop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

