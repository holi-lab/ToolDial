import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def simple_get(accountnumber: int, sortcode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can lookup any bank account details in Nigeria using our free bank API in few seconds to build applications faster.
		
		[BANK CODE ](https://gist.github.com/donejeh/5dd73ca4e2c8c94527219af52a5f53b8)
		[BANK LIST](https://gist.github.com/donejeh/591f2739d986d7ae6338ea2921d03cf4")"
    sortcode: [BANK CODE](https://gist.github.com/donejeh/5dd73ca4e2c8c94527219af52a5f53b8)
        
    """
    url = f"https://nigeria-bank-account-validation.p.rapidapi.com/"
    querystring = {'accountNumber': accountnumber, 'sortCode': sortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-bank-account-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

