import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wmcasinofood(wmcasino: str='one', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://www.foodservicemonthly.com/ Wm Casino là sảnh game đánh bài live casino nổi tiếng và phổ biến hàng đầu hiện nay tại các nhà cái online ở Việt Nam."
    
    """
    url = f"https://wmcasinofood.p.rapidapi.com/"
    querystring = {}
    if wmcasino:
        querystring['wmcasino'] = wmcasino
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wmcasinofood.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

