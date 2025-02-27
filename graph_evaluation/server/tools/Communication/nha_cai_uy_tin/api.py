import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nh_c_i_uy_t_n(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nhà cái uy tín - nhacai.la là trang chuyên review các nhà cái xanh chín nhất tại Việt Nam. Chơi cá cược thể thao, casino, tặng tiền cược miễn phí...
		Thông tin chi tiết:
		Website: [https://nhacai.la/](url)
		Địa chỉ: 400 Nguyễn Kim Cương, Tân Thạnh Tây, Củ Chi, Thành phố Hồ Chí Minh, Việt Nam 
		Email: vodathoa5085@gmail.com
		#nhacaiuytin #nhacai_la #topnhacaiuytin 
		Social:
		[https://twitter.com/nhacaila1
		](url)
		[https://www.linkedin.com/in/nhacaila/
		](url)
		[https://www.pinterest.com/nhacaila/
		](url)
		[https://social.msdn.microsoft.com/Profile/nhacaila
		](url)
		[https://social.technet.microsoft.com/Profile/nhacaila
		](url)
		[https://vimeo.com/nhacaila](url)"
    
    """
    url = f"https://nha-cai-uy-tin1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nha-cai-uy-tin1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nhacaila(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nhà cái uy tín - nhacai.la là trang chuyên review các nhà cái xanh chín nhất tại Việt Nam. Chơi cá cược thể thao, casino, tặng tiền cược miễn phí...
		Thông tin chi tiết:
		Website: [https://nhacai.la/](url)
		Địa chỉ: 400 Nguyễn Kim Cương, Tân Thạnh Tây, Củ Chi, Thành phố Hồ Chí Minh, Việt Nam 
		Email: vodathoa5085@gmail.com
		#nhacaiuytin #nhacai_la #topnhacaiuytin 
		Social:
		[https://twitter.com/nhacaila1
		](url)
		[https://www.linkedin.com/in/nhacaila/
		](url)
		[https://www.pinterest.com/nhacaila/
		](url)
		[https://social.msdn.microsoft.com/Profile/nhacaila
		](url)
		[https://social.technet.microsoft.com/Profile/nhacaila
		](url)
		[https://vimeo.com/nhacaila](url)"
    
    """
    url = f"https://nha-cai-uy-tin1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nha-cai-uy-tin1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

