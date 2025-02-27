import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mailcheck(email: str, domain: str='gmail.com, gmail.com, outlook.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "☑ Filter domain
		☑ Not start with a special character and must begin with a letter, 
		digit, or certain special characters like +, _, ., -.
		☑ Followed by the @ symbol to separate the username part and the domain part of the email address.
		☑ Domain part does not start or end with a dot or hyphen and only contains alphanumeric characters or dots.
		☑ Email string ends after the domain part and there are no additional characters."
    domain: Filter domain
Multiple domains can be entered, separated by commas.
Ex: gmail.com, example.com, yahoo.com
        
    """
    url = f"https://e-mail-check-invalid.p.rapidapi.com/WebAPIs/mail/emailValidator"
    querystring = {'email': email, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-mail-check-invalid.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

