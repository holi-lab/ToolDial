import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_info(keyword: str, level: str='district', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The data includes: the name of the province, city, and street, latitude and longitude, postcode, area code.
		The level can control the accuracy range of data query, the optional values are: province, city, district, street,default value is 'city'. The range increases step by step.
		**Query keyword must be in Chinese**"
    
    """
    url = f"https://chinese-newest-citys-data.p.rapidapi.com/api/area/cn"
    querystring = {'keyword': keyword, }
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-newest-citys-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

