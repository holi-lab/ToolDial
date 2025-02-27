import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getphoneintel(phone: str, country_hint: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trestle Phone Validation API validates the phone number and provides phone metadata such as carrier name, line type (landline, mobile, non-fixed voip, etc.), is the phone prepaid, etc. Typical use cases where the API is used include lead quality assessment and thus lead prioritization, as well as to weed out potential fraud, including bots using non-fixed voip or burner phones."
    phone: The phone number in E.164 or local format. The default country calling code is +1 (USA).
        country_hint: The ISO-3166 alpha-2 country code of the phone number. See: [ISO-3166](https://www.nationsonline.org/oneworld/country_code_list.htm)
test
        
    """
    url = f"https://phone-intelligence-api.p.rapidapi.com/3.0/phone_intel"
    querystring = {'phone': phone, }
    if country_hint:
        querystring['country_hint'] = country_hint
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-intelligence-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

