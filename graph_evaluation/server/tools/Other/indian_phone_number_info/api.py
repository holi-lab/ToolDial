import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getphoneinfo_mobile_number_the_first_four_digits_of_the_mobile_number(the_first_four_digits_of_the_mobile_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://avneesh-indian-phone-number-info-v1.p.rapidapi.com/getphoneinfo/?mobile_number={the_first_four_digits_of_the_mobile_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "avneesh-indian-phone-number-info-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

