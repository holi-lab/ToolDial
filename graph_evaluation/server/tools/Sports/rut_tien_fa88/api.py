import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fa88_link_t_i_app_fa88_online_android_ios_apk_mi_n_ph(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nếu bạn là một dân chơi cá cược chuyên nghiệp thì chắc hẳn đã biết sơ lược về nhà cái Fa88 online. Nhưng đối với những người chơi mới gia nhập thì vẫn còn là một địa chỉ khá mới mẻ. Nhưng dù là chuyên gia hay tân binh thì cần quan tâm những điều sau về nhà cái Fa88 online. Cùng tìm hiểu chi tiết qua bài viết sau."
    
    """
    url = f"https://rut-tien-fa88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rut-tien-fa88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

