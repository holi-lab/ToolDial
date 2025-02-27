import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trawex_car_rental(carrental: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "At Trawex, we have the best travel API solutions that can customize the car API as per your business location and customer type. We will adapt our technologies to fit your brand and will provide all the support to integrate your website with APIs. Our aim is to provide a new dimension of service concept in the travel industry and always strive to stay at the technological forefront by offering the best solutions."
    
    """
    url = f"https://trawex-car-rental.p.rapidapi.com/{carrental}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trawex-car-rental.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

