import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nivito(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Een keuken is geen keuken zonder een goede kraan. En als het om kranen gaat, is er geen betere dan de [Nivito](https://www.nivito.nl) Keukenkraan. Dit prachtige stuk machine is ontworpen voor moderne huizen en appartementen. Het heeft een strak design dat past bij elk aanrecht in de keuken. Bovendien is het gemaakt van hoogwaardige materialen die jarenlang meegaan.
		
		De Nivito Keukenkraan wordt geleverd met een enkele handgreep en is daardoor gemakkelijk in gebruik. Met deze handgreep zijn de waterdruk en temperatuur eenvoudig te regelen. De uitloop is ook zo ontworpen dat deze gemakkelijk kan worden gemanoeuvreerd. Zo bereik je alle hoeken en gaten van je gootsteen."
    
    """
    url = f"https://nivito2.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nivito2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

