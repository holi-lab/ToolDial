import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_address_verification_api(emailaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email is one of the most underrated customer engagement channels. Many email marketers, sales professionals and customer care teams struggle with the quality of their customer email ids. As a result, businesses are losing sales and marketing opportunities. 
		
		One of the common reasons is the poor email validation check during account signup or lead generation stage. Many form validation methods only check the syntax of the email address. But, email verification requires more than a syntax check. 
		
		Our Email Verification API is designed to render fast and straightforward email verification checks.
		
		The API provides the following validation:
		
		- Full standard compliance syntax check
		- Full domain part check, including mail servers configuration check
		- Checking against knowing abusive email domains and accounts list
		- Check if the email address is disposable or not
		 
		
		You can learn more about our process and get a deeper understanding of email verification methods in our blog post - [Deep dive into the real-time email address verification process and why we decided to do it differently?](https://www.bigdatacloud.com/blog/deep-dive-into-the-real-time-email-address-verification-process)"
    
    """
    url = f"https://email-address-verification1.p.rapidapi.com/data/email-verify"
    querystring = {'emailAddress': emailaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-address-verification1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

