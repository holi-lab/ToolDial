import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_email_data(getemail: bool, md5: str, plain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Email Data for get emails"
    
    """
    url = f"https://disposable-mail.p.rapidapi.com/getEmailTemp/"
    querystring = {'getEmail': getemail, 'md5': md5, 'plain': plain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disposable-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_temporary_mail_data(mail: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Temporary eMail Data"
    
    """
    url = f"https://disposable-mail.p.rapidapi.com/getEmailTemp/"
    querystring = {'mail': mail, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disposable-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

