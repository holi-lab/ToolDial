import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aba_behavioral_therapy_psych_associates_of_maryland(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[ABA Behavioral Therapy](https://www.pamllc.us/pam-services/aba-therapy) provided by Psych Associates of Maryland is known to improve attention, focus, social skills and memory. Applied Behavior Analysis has proved to be an effective treatment for autism and behavior modification."
    
    """
    url = f"https://psych-associates-of-maryland.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "psych-associates-of-maryland.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

