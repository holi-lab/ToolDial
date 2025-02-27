import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zeno_api(zenomedia: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Empowering App Developers to make their Apps even more Engaging
		Zeno Media has world class radio stations from around the world. Now any app can easily build and integrate our stations into your app through the brand new Zeno API program. Whether that be a radio app looking for more stations or a news app looking to give your users a new way to engage. With the new Zeno API anyone can incorporate over 50,000 stations in minutes. Contact subscription@zenomedia.com for trial API Keys and documentation."
    
    """
    url = f"https://zeno-media1.p.rapidapi.com/"
    querystring = {}
    if zenomedia:
        querystring['ZenoMedia'] = zenomedia
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zeno-media1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

