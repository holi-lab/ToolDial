import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_delivery_status(accountid: str, subaccountid: str, password: str, umid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API should be used to retrieve the current delivery status of a message sent using Wavecell."
    accountid: Your Wavecell Accountid
        subaccountid: Your Wavecell subaccountid
        password: Your Wavecell password
        umid: The Unique Message ID of the SMS for which you want to retrieve the delivery status
        
    """
    url = f"https://wavecell.p.rapidapi.com/getDLRApi.asmx/SMSDLR"
    querystring = {'AccountID': accountid, 'Subaccountid': subaccountid, 'Password': password, 'umid': umid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wavecell.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

