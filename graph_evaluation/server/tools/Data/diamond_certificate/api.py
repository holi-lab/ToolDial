import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def certificate(reportnumber: str, lab: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get diamond grading report along with PDF file link for GIA, IGI and HRD"
    
    """
    url = f"https://diamond-certificate.p.rapidapi.com/report"
    querystring = {'reportNumber': reportnumber, 'lab': lab, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diamond-certificate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

