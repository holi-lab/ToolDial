import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def venuedetail(vivacity_subscription: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A **multi-language** api
		
		Get venue details using KV (need resync to get latest data) and filter dynamic multi language data based on query params (eg. ?lang=zh-hans)"
    vivacity_subscription: subscription key value from api-gateway
        id: id of the venue
        
    """
    url = f"https://testing-docu.p.rapidapi.com/venue-i18n/venues/{is_id}/details"
    querystring = {'vivacity-subscription': vivacity_subscription, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menuorder(ocp_apim_subscription_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This api is still on progress
		
		Get information about the order from the menu."
    ocp_apim_subscription_key: subscription key value from api-gateway
        
    """
    url = f"https://testing-docu.p.rapidapi.com/test/instore/menu_order"
    querystring = {'Ocp-Apim-Subscription-Key': ocp_apim_subscription_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wechatuserinfo(ocp_apim_subscription_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Wechat user profile information using code or token provided from the Wechat backend/system upon scanning QR Code from Wechat app.  
		
		&nbsp;  
		
		> <a href="red">**NOTE**:</a> 
		>    - Authentication code/token will expire in 5minutes.
		>    - After generating a QR code it will automatically redirects you to the menu app and the code will be invalid once used. Here in the example it responded an invalid code because this is the second time we use the code.
		
		&nbsp;  
		
		Here is the sample of a successful response.
		<pre>
		{
		    "subscribe": 1, 
		    "openid": "o6_bmjrPTlm6_2sgVt7hMZOPfL2M", 
		    "nickname": "Band", 
		    "sex": 1, 
		    "language": "zh_CN", 
		    "city": "Guangzhou", 
		    "province": "Guangdong", 
		    "country": "China", 
		    "headimgurl":"http://thirdwx.qlogo.cn/mmopen/g3MonUZtNHkdmzicIlibx6iaFqAc56vxLSUfpb6n5WKSYVY0ChQKkiaJSgQ1dZuTOgvLLrhJbERQQ4eMsv84eavHiaiceqxibJxCfHe/0",
		    "subscribe_time": 1382694957,
		    "unionid": " o6_bmasdasdsad6_2sgVt7hMZOPfL"
		    "remark": "",
		    "groupid": 0,
		    "tagid_list":[128,2],
		    "subscribe_scene": "ADD_SCENE_QR_CODE",
		    "qr_scene": 98765,
		    "qr_scene_str": ""
		}
		</pre>"
    ocp_apim_subscription_key: subscription key value from api-gateway
        
    """
    url = f"https://testing-docu.p.rapidapi.com/test/instore/wechat/getUserInformation"
    querystring = {'Ocp-Apim-Subscription-Key': ocp_apim_subscription_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menudetail(vivacity_subscription: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A **multi-language** api
		
		Get menu details using KV (need resync to get latest data) and filter dynamic multi language data based on query params (eg. ?lang=zh-hans)"
    vivacity_subscription: subscription key value from api-gateway
        id: id of the menu
        
    """
    url = f"https://testing-docu.p.rapidapi.com/venue-i18n/menus/{is_id}/details"
    querystring = {'vivacity-subscription': vivacity_subscription, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venuerecommendation(ocp_apim_subscription_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return all the recommended dishes for the venue with their food names & details , meals by category, drinks and promotions."
    ocp_apim_subscription_key: subscription key value from api-gateway
        id: id of the venue
        
    """
    url = f"https://testing-docu.p.rapidapi.com/test/instore/v1/venue/{is_id}/recommendations"
    querystring = {'Ocp-Apim-Subscription-Key': ocp_apim_subscription_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def b2borganisationdetails(vivacity_subscription: str, org_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a specific organisation details with city list assigned."
    vivacity_subscription: subscription key value from api-gateway
        org_id: id of an organization
        
    """
    url = f"https://testing-docu.p.rapidapi.com/b2b/organizations/{org_id}"
    querystring = {'vivacity-subscription': vivacity_subscription, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-docu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

