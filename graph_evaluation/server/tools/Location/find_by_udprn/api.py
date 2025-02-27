import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_by_udprn(udprn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "“Find By UDPRN” API gets address for the specified UDPRN.
		
		UDPRN stands for ‘Unique Delivery Point Reference Number. A UDPRN is a unique numeric code (e.g. 64983) for any premise on the Postcode Address File.
		
		This API uses the latest PAF and Multiple Residence data from Royal Mail.UDPRN are unique identifiers for every address in the UK that Royal Mail has in its database."
    
    """
    url = f"https://find-by-udprn.p.rapidapi.com/v1/udprn/{udprn}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "find-by-udprn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

