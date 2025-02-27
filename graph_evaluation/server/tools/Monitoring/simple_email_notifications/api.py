import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_email_notification(email: str, subject: str='Cronjob-finished', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Intended usage is as a one liner. For example like this:
		backup.sh && curl https://simple-email-notifications.p.rapidapi.com/Backup finished ✅ /?email=demo@example.com&rapidapi-key=<your-rapidapi-key>
		
		We recommend to use your rapidapi key as a query parameter as to reduce the complexity of the curl call.
		
		Send a email notification to the specified recipient. Note that it's not possible to set a email body. It's only possible to set a subject, as this API is only supposed to be used as a notification tool for jobs or tasks that have finished or experienced errors. If you don't set a subject, "📩" will be set automatically for you."
    email: The E-Mail address that should receive this notification. Please note that it is NOT ALLOWED TO USE THIS API TO SEND ANY KIND OF MARKETING E-MAILS OR ANY E-MAIL COMMUNICATION WITH PARTIES THAT HAVE NOT GIVEN THEIR CONSENT TO RECEIVE THOSE E-MAILS! 
        subject: The subject to set for the E-Mail. In case you don't set a subject \\\\\\\\\\\\\\\"📩\\\\\\\\\\\\\\\" will be set for you. The maximum allowed length is 50 characters.
        
    """
    url = f"https://simple-email-notifications.p.rapidapi.com/{subject}"
    querystring = {'email': email, }
    if subject:
        querystring['subject'] = subject
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-email-notifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

