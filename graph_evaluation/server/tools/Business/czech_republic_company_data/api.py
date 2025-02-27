import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def companybyid(ico: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Czech company from it's ICO (Company ID)"
    
    """
    url = f"https://czech-republic-company-data.p.rapidapi.com/cgi-bin/ares/darv_std.cgi"
    querystring = {'ico': ico, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "czech-republic-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verbosecompanydetailsbyid(xml: int, ico: int, ver: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company data, including ownership information from a Czech company ID (ICO)"
    
    """
    url = f"https://czech-republic-company-data.p.rapidapi.com/cgi-bin/ares/darv_or.cgi"
    querystring = {'xml': xml, 'ico': ico, 'ver': ver, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "czech-republic-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

