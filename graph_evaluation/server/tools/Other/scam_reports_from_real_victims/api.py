import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scam_reports_from_real_victims_get_help_if_scammed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Scam Reports is an online help community that focuses on helping the victims of fraud by offering several scam reports and complete assistance in recovering funds.
		"The scammers always look for new tactics to fraud the people. Every day, lots of innocent people fall into this trap of scam brokers and firms, losing their funds.
		Scammed by a fraud broker? All Scam Reports is an online help community that offers several scam reports and assists you in the forex scam recovery and other scam recoveries such as Crypto scams, Bitcoin scams, Romance scams, etc. We work dedicatedly to provide you with the scam reports of different brokers and scammers, actively operating in the market."
		
		[https://allscamreports.com/](url)"
    
    """
    url = f"https://scam-reports-from-real-victims.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scam-reports-from-real-victims.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

