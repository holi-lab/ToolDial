import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def x2y2_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "x2y2 API endpoint (https://api.x2y2.io)
		Make over 1000 requests / minute
		
		Simply copy the path into the x2y2_path header.
		
		![](https://i.ibb.co/WKYfTDy/x2y2-path-get-header.png)"
    
    """
    url = f"https://x2y2.p.rapidapi.com/uniapi/xy3/v1/offer/list?contractAddress=0x34d85c9CDeB23FA97cb08333b511ac86E1C4E258&tokenId=28807&duration=0&page=1&pageSize=10&sort=desc&order=createdAt&isSufficient=1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "x2y2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

