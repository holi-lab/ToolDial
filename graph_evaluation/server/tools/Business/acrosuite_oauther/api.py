import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def openid_connecter(oauthcode: str=None, redirecthost: str=None, redirectport: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "OAuth/OpenID Connect(OIDC) authentication information from the third-party providers(**Google/LINE/YahooJP/Azure**) will be available. 
		After calling the API, you will get an OIDC list without **oAuthCode** or OIDC authentication information with **oAuthCode**.
		When you get an OIDC list, you may choose an **OAuthURL** to call it, then you will get OIDC authentication information.
		And it is not necessary to input redirectHost/redirectPort parameter here.
		Please refer to the sample[**OIDC Authentication**] in the following demo page.
		https://mars.restgate.net/intro/en/suite_en.html#oauther
		
		サードパティプロバイダー(**Google/LINE/YahooJP/Azure**)からOAuth **OpenID Connect(OIDC)**認証情報を取得します。
		このAPIを呼び出したら、OIDC一覧(**oAuthCode**付きなしの場合)又は認証情報(**oAuthCode**付きの場合)が戻られます。
		OIDC一覧が戻された場合は、続いて**OAuthURL**を選んで呼び出すと、該当する認証情報を取得できます。
		またここでは、redirectHost/redirectPortのパラメータを無視してください。
		実際の挙動は、下記のデモページに呼び出しサンプル[**OIDC認証**]をご参考ください。
		https://mars.restgate.net/intro/jp/suite_jp.html#oauther
		<hr>
		The following external document is for directly calling AcroSuite original APIs. Some Auth parameters may be neglected here.
		https://mars.acrochannel.com:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=2&withTitle=true&apiCode=02301016
		
		下記の外部ドキュメントは、AcroSuiteオリジナルAPIに対する資料であり、ここでは一部の認証パラメータを省略します。
		https://mars.acrochannel.com:8443/AcroSuite/v0_1/CstService/apiInfo/doc?appId=10006&language=1&withTitle=true&apiCode=02301016"
    oauthcode: OIDCProviderCode(GoogleOIDC/LineOIDC/YahooJPOIDC)
        
    """
    url = f"https://acrosuite-oauther.p.rapidapi.com/v0_1/CstService/tools/oauth/oidclist"
    querystring = {}
    if oauthcode:
        querystring['oAuthCode'] = oauthcode
    if redirecthost:
        querystring['redirectHost'] = redirecthost
    if redirectport:
        querystring['redirectPort'] = redirectport
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acrosuite-oauther.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

