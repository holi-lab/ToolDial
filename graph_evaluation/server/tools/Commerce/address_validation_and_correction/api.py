import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_validation_by_ubercx(request_id: str, addressline1: str, addressline2: str=None, city: str=None, state: str=None, zipcode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This GET request will return JSON response with corrected addresses, as applicable. If no matching address found, then matchCode is AVS_03. If it is perfect match, then matchCode is AVS01. If it's partial match with other possibilities, then matchCode is AVS_02. See detailing listing here https://snapcx.readme.io/docs/avsv1validateaddress"
    request_id: Any random unique string. Response returns this string as part of response. It helps in identifying unique requests.
        addressline1: Address Line 1
        addressline2: Optional second line of address.
        zipcode: 5 Digit zipcode of US Address. If either of city or state is missing, then this parameter is required.
        
    """
    url = f"https://snapcx-avs-v1.p.rapidapi.com/validateAddress?request_id={request_id}&street={addressline1}&state={state}&zipcode={zipcode}&city={city}"
    querystring = {}
    if addressline2:
        querystring['addressline2'] = addressline2
    if city:
        querystring['city'] = city
    if state:
        querystring['state'] = state
    if zipcode:
        querystring['zipcode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snapcx-avs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

