import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_verify(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "#### Verify email address
		*We do bill only valid or invalid verification statuses.*
		Pass the email as a GET parameter. Response properties description:
		```json
		{
		"status": "valid", // verification status of the email
		"reason": "accepted_email", // the reason of the status
		"email": "example@gmail.com",  // normalized email address
		"user": "example",  // The part before at-sign
		"domain": "gmail.com", // The part after at-sign
		"public": true, // Is address belongs to a publicly accessible email provider
		"disposable": false,  // Is address belongs to a disposable email service
		"role": false // Is address role based (e.g. abuse, admin, no-reply etc.)
		}
		```
		Possible verification statuses:
		-  `valid` - email address has been accepted by the mail server (safe to send)
		- `invalid` - email address is invalid, various reasons possible, see below (do not send)
		- `accept_all` - mail server accepts any email address (not recommended to send)
		- `unknown` - unable to verify email address (not recommended to send)
		
		Possible status reasons:
		- `accepted_email` - email address has been accepted by the mail server
		- `rejected_email` - email address has been rejected by the mail server
		- `invalid_syntax` - syntax of the email address is invalid according to RFC
		- `invalid_domain` - domain name of the email address does not exist
		- `no_mx_record` - mx record for the domain does not exist
		- `invalid_mx_record` - mx record of the domain is invalid
		- `dns_error` - error occurred while performing dns resolving of the domain
		- `smtp_error` - error occurred while performing smtp validation"
    email: Email address to validate
        
    """
    url = f"https://validect-email-verification-v1.p.rapidapi.com/v1/verify"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validect-email-verification-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

