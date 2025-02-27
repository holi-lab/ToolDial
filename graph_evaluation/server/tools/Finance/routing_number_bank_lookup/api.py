import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_bank_info(routingnumber: str, format: str='json', paymenttype: str='ach', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns information about a bank by passing in the routing number.
		
		Query Params:
		- **format:** defines the response type, currently XML and JSON are supported. Default value is JSON. Possible values: "xml", "json"
		- **paymentType:** choose to retrieve either the ACH or Wire Transfer information from the bank. Default value is ACH. Possible values: "ach", "wire"
		
		####Example:
		GET https://routing-number-bank-lookup.p.rapidapi.com/api/v1/121000248?paymentType=wire&format=json returns the wire transfer information for the bank corresponding to routing number 121000248 in json format."
    routingnumber: The routing number that you want to lookup
        format: Defines the format of the response. Default value is json. Possible inputs are 'xml' and 'json'
        paymenttype: Choose to retrieve either the ACH or Wire Transfer information from the bank. Default value is ACH. Possible values: 'ach' and 'wire'
        
    """
    url = f"https://routing-number-bank-lookup.p.rapidapi.com/api/v1/{routingnumber}"
    querystring = {}
    if format:
        querystring['format'] = format
    if paymenttype:
        querystring['paymentType'] = paymenttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "routing-number-bank-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

