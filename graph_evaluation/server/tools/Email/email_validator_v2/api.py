import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_validate(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate single email in real-time.
		
		**7 levels of verification:**
		
		1. By RFC standards.
		2. By service email (no-reply@, no-spam@, ...).
		3. MX records exist.
		4. By disposable/temporary email service.
		5. By spam catchers (like abusix.com network members).
		6. Connects to email service via standard protocol (SMTP) to check account exists.
		7. Connects to email service via internal api to check account banned or blocked (supports: Gmail, Mail.Ru, iCloud, Hotmail, Yahoo, Outlook, Rambler, Yandex and more)."
    email: Email for validation
        
    """
    url = f"https://email-validator18.p.rapidapi.com/email/validate"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-validator18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

