import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def basic_registered_blf_data_for_any_siblingauto_app_development(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the basic registered BLF data for any siblingauto APP development.
		
		You can provide query services to the public to view statuses of any data base attribute that was successfully formated by DBEIC bussines logic technology so as to provide special tracking services functionalities as explained in the siblingauto/Wamoja documentation.
		
		......."
    
    """
    url = f"https://wamoja.p.rapidapi.com/get_all_siblingauto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wamoja.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

