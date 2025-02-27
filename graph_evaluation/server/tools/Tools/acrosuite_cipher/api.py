import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_rsa_key(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Public and Private key pair of RSA.
		**RSA公開と秘密鍵**を生成するAPIです。
		<hr>
		The following external document is for directly calling AcroSuite original APIs. Some Auth parameters may be neglected here.
		https://mars.restgate.net:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=2&withTitle=true&apiCode=02301008
		
		下記の外部ドキュメントは、AcroSuiteオリジナルAPIに対する資料であり、ここでは一部の認証パラメータを省略します。
		https://mars.restgate.net:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=1&withTitle=true&apiCode=02301008"
    
    """
    url = f"https://acrosuite-cipher.p.rapidapi.com/v0_1/CstService/tools/cipher/rsakey"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acrosuite-cipher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

