import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_suggested_addresses_for_auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of suggested addresses which best match the supplied partial address.  Typically this endpoint is called as the user types an address on a form.   The suggested addresses are then displayed for the user to select the actual correct address."
    query: Specifies the partial address you wish to query.  Typically this is the value entered in a text input field from the UI.  The returned suggested addresses are what the API thinks are the most likely addresses the user will select based on this query parameter.  

Royal Mail postcodes are recognised as a special case, in which case the suggested addresses will be all within that postcode.

The query value is case insensitive and expected to be url encoded.
        
    """
    url = f"https://uk-address-and-postcodes.p.rapidapi.com/rapidapi/v1/autocomplete/addresses"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-address-and-postcodes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_the_full_address_of_a_suggested_address(addressid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the full address given an `addressId` from an auto completion suggested address.  Call this API to get the full address used to auto populate an address form when a user selects a suggested address."
    addressid: This is the `addressId` from a suggested address return from the `Fetch suggested addresses for auto complete` endpoint.

        
    """
    url = f"https://uk-address-and-postcodes.p.rapidapi.com/rapidapi/v1/address/{addressid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-address-and-postcodes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

