import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_multiple_users(email: str=None, orderby: str=None, term: str=None, offset: str=None, page: str=None, limit: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on all or multiple users in your Enterprise Hub. Use the optional parameters to return a more narrow range of users.The information returned for each user includes userId , status, email, lastActive, and more. "
    email: Return users with specified email(s).
        term: Specify users by text search, ID, or email.
        username: Return users with specified username(s).
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/users"
    querystring = {}
    if email:
        querystring['email'] = email
    if orderby:
        querystring['orderBy'] = orderby
    if term:
        querystring['term'] = term
    if offset:
        querystring['offset'] = offset
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_billing_plan_version(apiid: str, billingplanid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Billing Plan Version"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/plans/{billingplanid}/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_roles(rolelevel: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Roles"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/roles"
    querystring = {'roleLevel': rolelevel, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_organization(organizationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specified Organization's information, including the associated `email`, the number of `seats`, the Organization `name`, the `status`, and more."
    organizationid: A number associated with a specific organization. You can get the organizationId from the "Organization" tab on the Admin Panel, or by using the "Get All Organizations" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations/{organizationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_organizations_environment_admin(limit: str=None, offset: str=None, status: str='ACTIVE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Organization information for all of the Organizations in your Enterprise Hub.Organization information includes the associated `email`, the number of `seats`, the Organization `name`, the `status`, and more."
    limit: The number of orgs returned.
        offset: The number of orgs to skip.
        status: return active Orgs, deleted Orgs, or all Orgs.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/organizations"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_apis(category: str=None, limit: str=None, ownerid: str=None, tags: str=None, name: str=None, page: str=None, visibility: str='PUBLIC', orderby: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information for all APIs, including the `id`, `name`, and `status`."
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis"
    querystring = {}
    if category:
        querystring['category'] = category
    if limit:
        querystring['limit'] = limit
    if ownerid:
        querystring['ownerId'] = ownerid
    if tags:
        querystring['tags'] = tags
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if visibility:
        querystring['visibility'] = visibility
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_subscription(apiid: str, subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specific subscription to a specified API. "
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        subscriptionid: A number associated with a specific API subscription. You can get this using the "Get Subscriptions For All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/subscriptions/{subscriptionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_teams(organizationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all of the Teams in a specific Organization, including the `name`, `status`, and `description` for each Team. "
    organizationid: A number associated with a specific organization. You can get the organizationId from the "Organization" tab on the Admin Panel, or by using the "Get All Organizations" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations/{organizationid}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_app(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specified app, including the `name`, `description`, and `thumbnail`."
    appid: A number associated with a specific App. You can get the API ID by using the "Get All Apps" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apps/{appid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_api_docs(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all of the docs associated with an API, including the docId."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/docs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_apps(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on all of the Apps associated with an account."
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apps"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_app_keys(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all App Keys for a specified App."
    appid: A number associated with a specific App. You can get the API ID by using the "Get All Apps" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apps/{appid}/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_transactions(category: str, page: int=None, orderby: str='createdAt', filters: str=None, limit: int=None, order: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Transactions"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/transactions"
    querystring = {'category': category, }
    if page:
        querystring['page'] = page
    if orderby:
        querystring['orderBy'] = orderby
    if filters:
        querystring['filters'] = filters
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_developers_for_a_billing_plan(billingplanid: str, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Developers for a Billing Plan"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/plans/{billingplanid}/developers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_api_developers(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Api Developers"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/developers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag_definition(tagdefinitionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on a specific Tag within your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/tagdefinitions/{tagdefinitionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_organization_audit_log(organizationid: str, limit: str=None, page: str=None, orderby: str=None, searchterm: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the audit trail for a specified Organization. Information in the audit trail includes details like `eventName`, `user`, `attributes` and more."
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations/{organizationid}/audit"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if orderby:
        querystring['orderBy'] = orderby
    if searchterm:
        querystring['searchTerm'] = searchterm
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_endpoints(versionid: str, apiid: str, showdeleted: bool=None, page: str=None, include: str=None, orderby: str=None, offset: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Endpoints"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions/{versionid}/endpoints"
    querystring = {}
    if showdeleted:
        querystring['showDeleted'] = showdeleted
    if page:
        querystring['page'] = page
    if include:
        querystring['include'] = include
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_endpoint(endpointid: str, versionid: str, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Endpoint"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions/{versionid}/endpoints/{endpointid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category(categoryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on a specific Category within your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/categories/{categoryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories(limit: str=None, page: str=None, name: str=None, offset: str=None, orderby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the Categories that exist in your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/categories"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    if offset:
        querystring['offset'] = offset
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_entity_roles(entityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Entity Roles"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/entities/{entityid}/roles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_organizations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Organization information for all of the Organizations you have access to. Organization information includes the associated `email`, the number of `seats`, the Organization `name`, the `status`, and more."
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_openapi(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specified API's OpenAPI file in REST format."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/openapi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_subscription(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info on an existing subscription "
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/subscriptions/{subscriptionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_user_subscriptions(limit: str=None, query: str=None, page: str=None, orderby: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all your subscriptions"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/subscriptions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    if page:
        querystring['page'] = page
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_collections(orderby: str=None, orgid: str=None, offset: str=None, limit: str=None, page: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all of the collections that exist within your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/collections"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if orgid:
        querystring['orgId'] = orgid
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on specified API, including `category`, `id`, and `name`"
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection(collectionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on a specific collection within your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/collections/{collectionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_log_details(calltime: str, apiid: str, requestid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get API logs details information, including `endpoint`, `http method`, `payload` and more."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        requestid: A number associated with a specific request. You can get the request ID from the "Get API Logs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/logs/{requestid}"
    querystring = {'callTime': calltime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_by_external_custom_id(externalcustomid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on specified API based on a given custom external ID, including `category`, `id`, and `name`"
    externalcustomid: An optional string associated with a specific API.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/external-custom-id/{externalcustomid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_api_versions(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the versions (and their corresponding information) that exist for a specified API."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def progress_of_api_version_file_upload(trackingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the progress of an API Version file upload process"
    trackingid: A string associated with a specific API version file upload process.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiId}/versions/async/summary/{trackingid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_public_apis(term: str=None, tags: str=None, orderby: str=None, offset: str=None, collectionids: str='[]', limit: str=None, page: str=None, categorynames: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all public APIs by specific terms."
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/search"
    querystring = {}
    if term:
        querystring['term'] = term
    if tags:
        querystring['tags'] = tags
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    if collectionids:
        querystring['collectionIds'] = collectionids
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if categorynames:
        querystring['categoryNames'] = categorynames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_version(apiid: str, versionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information for a specified version of an API, including the `id`, `name`, and `status`."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        versionid: A string associated with a specific API version. You can get the Version ID by using the "Get API Version" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions/{versionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_requesting_user_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed user information that is associated to the account"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_subscriptions(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all of the subscriptions to a specified API."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/subscriptions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_users(organizationid: str, organizationteamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all of the users that belong to a specific team. Returned user information includes the user `id`, `email`, `username`, and `lastActive`. "
    organizationid: A number associated with a specific organization. You can get the organizationId from the "Organization" tab on the Admin Panel, or by using the "Get All Organizations" endpoint.
        organizationteamid: A number associated with a specific team in an organization. You can get the organizationTeamId by using the "Get All Teams" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations/{organizationid}/teams/{organizationteamid}/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quota_export(month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download Quota Export file for a specific month"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/quotas/export/{year}/{month}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_teams(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the `organizationTeamIds` for each Team the user is a member of."
    userid: A number associated with a specific user. You can use the "Get All Users" endpoint to get the userId, or obtain it from the admin panel.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/users/{userid}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_spotlights(showunpublished: bool, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get API Spotlights"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/spotlights"
    querystring = {'showUnpublished': showunpublished, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_user(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on a specified user, including the `email`, `name`, and `thumbnail`. The endpoint also indicates if the user has two factor authentication methods enabled."
    userid: A number associated with a specific user. You can use the "Get All Users" endpoint to get the userId, or obtain it from the admin panel.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/users/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_api_tags(apiid: str, orderby: str=None, limit: str=None, offset: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all of the tags associated with an API, including the `tagId`."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/tags"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_tag_definitions(isvisible: bool=None, requiredonapi: bool=None, showtagname: bool=None, editablebyprovider: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on all the Tags within your environment"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/tagdefinitions"
    querystring = {}
    if isvisible:
        querystring['isVisible'] = isvisible
    if requiredonapi:
        querystring['requiredOnAPI'] = requiredonapi
    if showtagname:
        querystring['showTagName'] = showtagname
    if editablebyprovider:
        querystring['editableByProvider'] = editablebyprovider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_gateways(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Gateways"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/gateways"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_logs(is_from: str, apiid: str, to: str, orderby: str='callTime', order: str='desc', filters: str=None, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the logs for a particular API."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/logs"
    querystring = {'from': is_from, 'to': to, }
    if orderby:
        querystring['orderBy'] = orderby
    if order:
        querystring['order'] = order
    if filters:
        querystring['filters'] = filters
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team(organizationid: str, organizationteamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specific Team, including the `name`, `status`, and `description`."
    organizationid: A number associated with a specific organization. You can get the organizationId from the "Organization" tab on the Admin Panel, or by using the "Get All Organizations" endpoint.
        organizationteamid: A number associated with a specific team in an organization. You can get the organizationTeamId by using the "Get All Teams" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/organizations/{organizationid}/teams/{organizationteamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_current_version(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information for the current version of a specified API, including the `id`, `name`, and `status`."
    apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_requesting_user_teams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the team ID's that are associated to your account"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/me/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_version_openapi(versionid: str, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specified API Version's OpenAPI file in REST format."
    versionid: A string associated with a specific API version. You can get the Version ID by using the "Get API Version" endpoint.
        apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/versions/{versionid}/openapi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_spotlight(apiid: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get API Spotlight"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/spotlights/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_tag(tagid: str, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tag information for a specific API"
    tagid: A number associated with a specific API Tag. You can get the Tag ID from the by using the "Get All API Tags" endpoint.
        apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/tags/{tagid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_audit_login(userid: str, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total number of logins and detailed login logs for a specified user. The logs include the `time`, `geography` , and other `attributes` associated with each login by the user. By default, the endpoint returns up to 10 logs. You can increase or decrease the amount of logs returned using the limit parameter."
    userid: A number associated with a specific user. You can use the "Get All Users" endpoint to get the userId, or obtain it from the admin panel.
        limit: The number of logs returned. If left unspecified, the default is 10 logs.
        offset: The number of logs to skip.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/admin/users/audit/{userid}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_version_openapi(versionid: str, apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specified API Version's OpenAPI file."
    versionid: A string associated with a specific API version. You can get the Version ID by using the "Get API Version" endpoint.
        apiid: A string associated with a specific API. You can get the API ID from the "APIs" tab on the Admin Panel, or by using the "Get All APIs" endpoint.
        
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/rapidapi-file/{apiid}/versions/{versionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_billing_plans(apiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Billing Plans"
    
    """
    url = f"https://openapi-provisioning.p.rapidapi.com/v1/apis/{apiid}/plans"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openapi-provisioning.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

