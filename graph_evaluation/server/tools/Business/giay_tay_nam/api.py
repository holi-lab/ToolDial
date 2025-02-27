import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def da_laforce(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Đồ da Laforce là thương hiệu uy tín tại Hà Nội cũng như trên toàn quốc về sản phẩm đồ da như giày, túi, ví, thắt lưng, găng tay."
    
    """
    url = f"https://giay-tay-nam.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giay-tay-nam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

