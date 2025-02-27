import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company(vanity_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Company Data"
    vanity_name: Company vanityName in Linkedin

https://www.linkedin.com/company/**(vanityName)**

Examples:
https://www.linkedin.com/company/microsoft --> microsoft
https://www.linkedin.com/company/adobe --> adobe
https://www.linkedin.com/company/att-> att

        
    """
    url = f"https://linkedin-companies-data.p.rapidapi.com/"
    querystring = {'vanity_name': vanity_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-companies-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

