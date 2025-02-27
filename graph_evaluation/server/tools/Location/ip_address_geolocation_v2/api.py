import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getiplocation(ip: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the location of an IP address (country, region, city, zipcode, latitude and longitude) and the associated timezone in JSON or XML format."
    ip: IP Address. It supports both IPv4 and IPv6 address format.
        format: It supports json and xml response format.
        
    """
    url = f"https://chrislim2888-ip-address-geolocation.p.rapidapi.com/"
    querystring = {'ip': ip, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrislim2888-ip-address-geolocation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

