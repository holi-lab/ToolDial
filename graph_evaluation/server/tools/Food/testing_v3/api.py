import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wechatuserinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
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
    
    """
    url = f"https://testing144.p.rapidapi.com/test/instore/wechat/getUserInformation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcategory(sort: str, where: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/settings"
    querystring = {'sort': sort, 'where': where, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findvenueorganisation(populate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-venues/api/organizations/5ed4ef97c318ff43e56edfd9"
    querystring = {'populate': populate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monitoring the api connectivity."
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findoneuserlocation(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " FindOne UserLocation"
    id: (Required) 
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-locations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcuisine(limit: int, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/cuisines"
    querystring = {'limit': limit, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findoneuserorganization(populate: str, is_id: str, select: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    populate: relationship join
        id: (Required) 
        select: field to respond
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-organizations/{is_id}"
    querystring = {'populate': populate, 'select': select, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validateemail(email: str, organization: str, account: str='5f6e05f6fa6ba2260568f7e0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When inviting a user this will first validate if the email input is valid and is not yet taken. Email must be unique."
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/auth/validate-email"
    querystring = {'email': email, 'organization': organization, }
    if account:
        querystring['account'] = account
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findorganisationcities(populate: str, limit: int, where: str, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-venues/api/organization-cities"
    querystring = {'populate': populate, 'limit': limit, 'where': where, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venuedetail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A **multi-language** api
		
		Get venue details using KV (need resync to get latest data) and filter dynamic multi language data based on query params (eg. ?lang=zh-hans)"
    id: id of the venue
        
    """
    url = f"https://testing144.p.rapidapi.com/venue-i18n/venues/{is_id}/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findoneuseractivity(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " FindOne UserActivity"
    id: (Required) 
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-activities/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menudetail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A **multi-language** api
		
		Get menu details using KV (need resync to get latest data) and filter dynamic multi language data based on query params (eg. ?lang=zh-hans)"
    id: id of the menu
        
    """
    url = f"https://testing144.p.rapidapi.com/venue-i18n/menus/{is_id}/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menuorder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This api is still on progress
		
		Get information about the order from the menu."
    
    """
    url = f"https://testing144.p.rapidapi.com/test/instore/menu_order"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findproduct(limit: int, populate: str, organization: str, page: int, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/products"
    querystring = {'limit': limit, 'populate': populate, 'organization': organization, 'page': page, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def b2borganisationdetails(org_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a specific organisation details with city list assigned."
    org_id: id of an organization
        
    """
    url = f"https://testing144.p.rapidapi.com/b2b/organizations/{org_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finduser(page: int, search: str, organization: str, skip: int, select: str, sort: str, limit: int, populate: str, where: str='{"email": "tangpuzehmz@gmail.com"}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Find User"
    page: used to retrieve the next/previous pages of content
        search: normal search query
        organization: an organization id from this api /api/organizations
        skip: the number of items to skip
        select: field to respond
        sort: sort via createdAt field in descending order(-1 Desc, 1 Asc)
        limit: this will limit the returned result count
        populate: relationship join
        where: Mongodb filter queries
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/users"
    querystring = {'page': page, 'search': search, 'organization': organization, 'skip': skip, 'select': select, 'sort': sort, 'limit': limit, 'populate': populate, }
    if where:
        querystring['where'] = where
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findoneuser(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " FindOne User"
    id: (Required) 
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findimage(sort: str, where: str, limit: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/images"
    querystring = {'sort': sort, 'where': where, 'limit': limit, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finduseractivity(search: str, where: str, sort: str, organization: str, page: int, skip: int, limit: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Find UserActivity"
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-activities"
    querystring = {'search': search, 'where': where, 'sort': sort, 'organization': organization, 'page': page, 'skip': skip, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finduserorganization(skip: int, populate: str, select: str, organization: str, limit: int, sort: str, page: int, where: str='{"role": 3}', search: str='{"role": 3}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    skip: the number of items to skip
        populate: relationship join
        select: field to respond
        organization: an organization id from this api /api/organizations
        limit: this will limit the returned result count
        sort: sort via createdAt field in descending order(-1 Desc, 1 Asc)
        page: used to retrieve the next/previous pages of content
        where: Mongodb filter queries
        search: normal search query
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-organizations"
    querystring = {'skip': skip, 'populate': populate, 'select': select, 'organization': organization, 'limit': limit, 'sort': sort, 'page': page, }
    if where:
        querystring['where'] = where
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findproductgroups(where: str, limit: int, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/product-groups"
    querystring = {'where': where, 'limit': limit, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcountries(page: int, sort: str, limit: int, populate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/countries"
    querystring = {'page': page, 'sort': sort, 'limit': limit, 'populate': populate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findaccount(page: int, limit: int, organization: str, sort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-venues/api/accounts"
    querystring = {'page': page, 'limit': limit, 'organization': organization, 'sort': sort, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venuerecommendation(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return all the recommended dishes for the venue with their food names & details , meals by category, drinks and promotions."
    id: id of the venue
        
    """
    url = f"https://testing144.p.rapidapi.com/test/instore/v1/venue/{is_id}/recommendations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findvenuecategories(populate: str, sort: str, organization: str, page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/venue-categories"
    querystring = {'populate': populate, 'sort': sort, 'organization': organization, 'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findtags(sort: str, page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/tags"
    querystring = {'sort': sort, 'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findonevenue(is_id: str, sort: str, page: int, populate: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " FindOne Venue"
    id: (Required) 
        
    """
    url = f"https://testing144.p.rapidapi.com/ms-venues/api/venues/{is_id}"
    querystring = {'sort': sort, 'page': page, 'populate': populate, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finduserlocation(search: str, page: int, organization: str, where: str, skip: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Find UserLocation"
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-securities/api/user-locations"
    querystring = {'search': search, 'page': page, 'organization': organization, 'where': where, 'skip': skip, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findpaymentstype(where: str, sort: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://testing144.p.rapidapi.com/ms-categories/api/payment-types"
    querystring = {'where': where, 'sort': sort, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing144.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

