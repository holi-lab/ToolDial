import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generatetabledata(tablekey: str, nrrecords: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates your test/demo/random data based on the definition you created for free on your account, on onlinedatagenerator.com website.  
		The tablekey parameter can be found on the mytables->edit one table"
    
    """
    url = f"https://onlinedatageneratorapi1.p.rapidapi.com/GenerateTableData"
    querystring = {'tablekey': tablekey, 'nrrecords': nrrecords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlinedatageneratorapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

