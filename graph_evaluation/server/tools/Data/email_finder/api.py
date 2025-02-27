import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_emails(name: str, email_domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the company emails of a professional by full name (John Doe) and email domain (@company.com)"
    name: The name of the person/professional to find. We recommend using the full name of the person for more accurate results.
        email_domain: The email domain of the company or person to search for.
        
    """
    url = f"https://email-finder4.p.rapidapi.com/find-emails"
    querystring = {'name': name, 'email_domain': email_domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

