import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def c_a_h_ng_n_i_th_t_m_y_tre(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nội thất Home Mây cung cấp các sản phẩm nội thất mây tre tại Đà Nẵng và trên toàn quốc. Cam kết sản phẩm chất lượng cao với mức giá ưu đãi nhất."
    
    """
    url = f"https://noithathomemay.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "noithathomemay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

