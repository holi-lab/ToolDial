import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ipix_lms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The software can build learning courses based on a singular approach or a combination of teaching styles that they find suitable to ensure that their trainees are learning and retaining what they learn, turning them into effective and productive assets of the company."
    
    """
    url = f"https://ipix-lms.p.rapidapi.comhttps://www.ipixtechnologies.com/learning-management-system.html"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipix-lms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

