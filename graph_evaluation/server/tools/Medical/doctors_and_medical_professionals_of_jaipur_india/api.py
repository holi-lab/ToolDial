import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jaipur_doctors_api(page_no: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of almost 11K doctors and medical professionals of Jaipur, India"
    page_no: Get the details of almost 11K doctors and medical professionals of Jaipur, India
        
    """
    url = f"https://doctors-and-medical-professionals-of-jaipur-india.p.rapidapi.com/dentist/{page_no}"
    querystring = {}
    if page_no:
        querystring['page_no'] = page_no
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doctors-and-medical-professionals-of-jaipur-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

