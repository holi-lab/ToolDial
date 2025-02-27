import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_email_of_a_person(first_name: str, domain: str, last_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get email of anyone in the internet. Best for lead generation, prospecting, and cold marketing."
    
    """
    url = f"https://email-finder8.p.rapidapi.com/fetch_email_of_person"
    querystring = {'first_name': first_name, 'domain': domain, 'last_name': last_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

