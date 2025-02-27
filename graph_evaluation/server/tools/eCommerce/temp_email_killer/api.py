import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_domain_email_mailbox_address(data: str, ip: str='154.115.9.195', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "So, below there will be a boring description of the API, while you have enthusiasm, remember the main thing: the main response parameter is "result":, if this parameter is true - feel free to skip this EMAIL for registration, if false, then take your time - tell the user "Register with temporary mails are prohibited by resource policy".
		
		Also, you have a trump card up your sleeve - this is the "error":, parameter: if everything goes well, then it is always 0, if the user makes mistakes, then this parameter takes numerical values, you can, if you wish, set him on the right path, prompting him to check the correctness of the EMAIL input"
    data: Domain or full Email address
        ip: Optional parameter. The IP address of the client that sends the request to your resource is required for a more detailed display of statistics
        
    """
    url = f"https://temp-email-killer.p.rapidapi.com/check"
    querystring = {'data': data, }
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temp-email-killer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

