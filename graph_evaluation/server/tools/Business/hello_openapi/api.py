import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getstoreextended(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/InfoExtended/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmerchant(account_id: str, partner_code: str=None, event_code: str='"done_sign_up"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    partner_code: the partner code, if this parameter is set then `account_id` is the partner `account ID` if `partner_code` is not set then `account_id` is the Movylo merchant account ID
        event_code: the event codes, if this parameter is set then `account_id` is the partner `account ID` if `partner_code` is not set then `account_id` is the Movylo merchant account ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Merchant/{account_id}/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    if event_code:
        querystring['event_code'] = event_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstores(search_string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    search_string: string to perform store search by name
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Stores/Search/"
    querystring = {}
    if search_string:
        querystring['search_string'] = search_string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstoreinfopages(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/InfoPages/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcustomerstats(account_id: int, store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the customer's stats"
    account_id: Id of the Store's customer account
        store_id: Id of the Movylo store where the customer is
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Customer/{account_id}/Stats/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcustomerloyaltypoints(account_id: int, store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get loyalty points of a customer by the store_id and the account_id"
    account_id: Id of the customer within the Movylo store
        store_id: Id of the Movylo store where the customer is
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Loyalty/Customer/{account_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstoreorders(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Orders/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getloyaltyreward(store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get loyalty rewards list"
    store_id: Id of the Movylo store where the customer is
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Loyalty/Rewards/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcustomers(store_id: int, account_id: int=None, search_string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get customers by the customer account_id or by a search string (on fields 'email', 'phone', 'first_name', 'last_name' and 'loyalty_code')"
    store_id: Id of the Movylo store where the customer is
        account_id: Id of the Store's customer account, if 'search_string' is not used only the customer with this account id will be returned otherwise this will be ignored and will be performed a search
        search_string: string to perform customer search, search is performed on fields 'email', 'phone', 'first_name', 'last_name' and 'loyalty_code' if this field is set will be performed a serach and the 'account_id' field will be ignored
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Customer/"
    querystring = {}
    if account_id:
        querystring['account_id'] = account_id
    if search_string:
        querystring['search_string'] = search_string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmoremerchant(account_id: str, event_code: str='"done_sign_up"', partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    event_code: the event codes, if this parameter is set then `account_id` is the partner `account ID` if `partner_code` is not set then `account_id` is the Movylo merchant account ID
        partner_code: the partner code, if this parameter is set then `account_id` is the partner `account ID` if `partner_code` is not set then `account_id` is the Movylo merchant account ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Merchant/"
    querystring = {'account_id': account_id, }
    if event_code:
        querystring['event_code'] = event_code
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcustomercoupons(account_id: int, store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the customer's coupons"
    account_id: Id of the Store's customer account
        store_id: Id of the Movylo store where the customer is
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Customer/{account_id}/Coupons/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcouponstatus(store_id: int, coupon_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a coupons status by the store_id and the coupon code"
    store_id: Id of the Movylo store where the customer is
        coupon_code: code of the coupon to check
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Coupon/{coupon_code}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstoreproducts(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Products/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstore(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcouponsstats(store_id: int, from_date: str=None, to_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the stats about coupons use for a time range"
    store_id: Id of the Movylo store to check
        from_date: in format 'yyyy-mm-dd', if not set returns last 30 days
        to_date: in format 'yyyy-mm-dd', if not set returns last 30 days
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Coupons/Stats/"
    querystring = {}
    if from_date:
        querystring['from_date'] = from_date
    if to_date:
        querystring['to_date'] = to_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreviews(store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Reviews"
    store_id: Id of the Movylo store to check
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Reviews/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplanbyid(plan_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get plan info by the plan ID"
    plan_id: Id of the plan
        partner_code: the partner code, if this parameter is set then 'plan_id' is the partner plan ID, if 'partner_code' is not set then 'plan_id' is the internal Movylo plan ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Billing/Plan/Id/{plan_id}/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplans(partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get plans info"
    partner_code: the partner code, if this parameter is set then returns plans associociated to the Partner
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Billing/Plans/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstoreoffers(store_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Id of the Store
        partner_code: the partner code, if this parameter is set then 'store_id' is the partner store ID, if 'partner_code' is not set then 'store_id' is the Movylo store ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Offers/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproduct(store_id: int, product_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get products by the product id"
    store_id: Id of the Movylo store where the customer is
        product_id: Id of the Store's product
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Product/{product_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfeedback(store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Feedback"
    store_id: Id of the Movylo store to check
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Feedback/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def merchant_stores_account_id(account_id: str, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account_id: Id of the Merchant account
        partner_code: the partner code, if this parameter is set then 'account_id' is the partner account ID, if 'partner_code' is not set then 'account_id' is the Movylo merchant account ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Merchant/Stores/{account_id}/"
    querystring = {}
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getautopilot(store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if the AutoPilot is enabled for the specified Store"
    store_id: Id of the Movylo store to check
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/MarketingAutomation/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcustomerprefstoredata(store_id: str, account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Store Home data for the customer"
    store_id: Id of the Movylo store where to search the customers
        account_id: Id of the Store's customer account
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Home/Customer/{account_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstorestats(store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the stats about the store"
    store_id: Id of the Movylo store to check
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Store/{store_id}/Stats/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def merchant_login_id(is_id: str, password: str=None, partner_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "hi"
    id: Can be the Id of the Merchant account or the registration email. If use the registration email you need to also set the password parameter.
        password: the password of the account, required if authenticate via email
        partner_code: the partner code, if this parameter is set then 'account_id' is the partner account ID, if 'partner_code' is not set then 'account_id' is the Movylo merchant account ID
        
    """
    url = f"https://hello-openapi.p.rapidapi.com/Merchant/Login/{is_id}/"
    querystring = {}
    if password:
        querystring['password'] = password
    if partner_code:
        querystring['partner_code'] = partner_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hello-openapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

