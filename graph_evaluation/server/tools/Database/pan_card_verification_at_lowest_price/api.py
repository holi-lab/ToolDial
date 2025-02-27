import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_pan_card(x_rapid_api: str, pannumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instant PAN Card Verification using government database check.
		
		This PAN Card Verification API instantly verifies details of a PAN Card by confirming them from the Government database.
		
		This makes your onboarding process faster, safer and smarter. With our PAN verification, you can be confident that the individuals or merchants you onboard hold a valid PAN card and have provided you with the right identity proof."
    
    """
    url = f"https://pan-card-verification-at-lowest-price.p.rapidapi.com/verifyPan/{pannumber}"
    querystring = {'x-rapid-api': x_rapid_api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pan-card-verification-at-lowest-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

