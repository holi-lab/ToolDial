import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trawex_cruise_api(cruise: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web API was created by Trawex. You can find the Trawex portal / hompage [here](https://www.trawex.com/cruise-api.php). If you need Trawex Cruise API support, you can contact support directly at contact@trawex.com, or reach out to their Twitter account at [@trawextech](https://twitter.com/trawextech). For more information, check out their [API Documentation](https://www.trawex.com/api-integration.php). The Trawex Cruise API is **not currently available on the RapidAPI marketplace**. Click "**Request this API on RapidAPI**" to let us know if you would like to access to this API. Meanwhile, you can check out the [top APIs that currently available](https://rapidapi.com/hub) for developers."
    cruise: Trawex is a well-known brand to offering best in class cruise API for travel agents by which they can book cruises online in an easy manner. Our API provides an interface that connects cruise suppliers and online travel agencies where online travel agents can view their real-time inventory and availability through cruise booking software. We include cruise providers on a single platform where travel agents can access through cruise API to do easy online booking of cruises.
        
    """
    url = f"https://cruise.p.rapidapi.com/"
    querystring = {'Cruise': cruise, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cruise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

