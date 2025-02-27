import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_graham_number(bookvalue: str, eps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API Get the Graham Number by takes into account earnings per share and book value per share.
		**Require Params**
		eps : Earning Per Share (TTM EPS ) 
		bookvalue : Book Value (MRQ Book Value per Share)"
    
    """
    url = f"https://graham-number.p.rapidapi.com/price"
    querystring = {'bookvalue': bookvalue, 'eps': eps, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "graham-number.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

