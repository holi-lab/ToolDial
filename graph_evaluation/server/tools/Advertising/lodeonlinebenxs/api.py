import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lodeonline(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://www.benxoso.com/  Lô đề online benxoso cập nhật top 10 trang đánh lô đề online uy tín nhất hiện nay với tỉ lệ ăn cao nhất, xanh chín, minh bạch, nạp rút nhanh, dễ đặt cược và nhiều khuyến mãi."
    
    """
    url = f"https://lodeonlinebenxs.p.rapidapi.com/lodeonlinebenxs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lodeonlinebenxs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

