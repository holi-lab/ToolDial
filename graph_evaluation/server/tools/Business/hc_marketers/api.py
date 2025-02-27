import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def doctors_email_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our Doctors Email List will help you connect with the top Doctors worldwide. Our list consists of medical specialists who are the best in diagnosing and treating heart disorders. With the correct Mailing Database, you can reach out to these specialists and collaborate with them in promoting your products and services. We can customize our Doctors Email Marketing List based on your core business competencies and agenda. Through our verified email database you will surely get more business leads, the best response to your marketing campaigns and increase your sales and ROI."
    
    """
    url = f"https://hc-marketers1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hc-marketers1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

