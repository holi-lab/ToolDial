import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_email_address(emails: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Do Validity Check for Syntax and Domain's MX records of MailAddress set.
		メールアドレスの**構文**と**ドメイン**を検証します。
		Query parameter
		**emails**: EmailAddress set(Multiple Emails is OK)/(複数可)
		<hr>
		The following external document is for directly calling AcroSuite original APIs. Some Auth parameters may be neglected here.
		https://mars.restgate.net:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=2&withTitle=true&apiCode=02301012
		
		下記の外部ドキュメントは、AcroSuiteオリジナルAPIに対する資料であり、ここでは一部の認証パラメータを省略します。
		https://mars.restgate.net:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=1&withTitle=true&apiCode=02301012"
    emails: EmailAddress set(Multi Emails is OK)/EmailAddressセット(複数可)
        
    """
    url = f"https://acrosuite-mailer.p.rapidapi.com/v0_1/CstService/tools/mail/isvalidaddress"
    querystring = {'emails': emails, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acrosuite-mailer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

