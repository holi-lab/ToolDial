import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_email_format(firstname: str, lastname: str, companyurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We find for you the format of email from first name, last name, company URL."
    companyurl: **DO NOT** add:
- www
- https://
        
    """
    url = f"https://company-email-format-finder.p.rapidapi.com/Find_Email_Format"
    querystring = {'firstName': firstname, 'lastName': lastname, 'companyURL': companyurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "company-email-format-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

