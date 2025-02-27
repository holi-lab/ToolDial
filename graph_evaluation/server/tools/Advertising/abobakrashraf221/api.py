import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def m1223saa(fathia: str='trtj', fgsjffh: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "سهولة الوصول إلينا"
    
    """
    url = f"https://abobakrashraf221.p.rapidapi.com/"
    querystring = {}
    if fathia:
        querystring['fathia'] = fathia
    if fgsjffh:
        querystring['fgsjffh'] = fgsjffh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abobakrashraf221.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

