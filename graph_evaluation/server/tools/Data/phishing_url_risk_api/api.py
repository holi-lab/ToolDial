import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phishing_url_detection_api(url: str='appleid.apple.com-sa.pm', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point takes a  'GET' request with urll/string as a parameter and retunrs  phishing risk indicators such as :  url Redirecting risk, SubDomains, HTTPS, Domain Regulation Length,
		IframeRedirection, Age of Domain, DNS Recording, WebsiteTraffic."
    
    """
    url = f"https://phishing-url-risk-api.p.rapidapi.com/url/"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phishing-url-risk-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

