import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web_development_company(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ipix Tech Services Pvt.Ltd, software development company headquartered in Dubai which specializes in Web Applications, Web development & Digital marketing to serve globally by combining wide capabilities & unparalleled experience of all businesses."
    
    """
    url = f"https://web-development-company.p.rapidapi.comhttps://www.ipixtechnologies.com/web-development.html"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-development-company.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

