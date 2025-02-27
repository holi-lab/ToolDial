import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def viewdb(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "등록된 **데이터베이스를 열람**합니다. 데이터베이스에 저장되어있는 모든 UID와 UID의 개수를 돌려줍니다.
		
		View the Database"
    
    """
    url = f"https://alchera-face-authentication.p.rapidapi.com/v1/face"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alchera-face-authentication.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

