import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_entity_users_entity_users_get(first_name: str=None, created_at: str=None, created_at_lt: str=None, pagination_token: str=None, sort: str='{
  "0": "u",
  "1": "p",
  "2": "d",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', created_at_gte: str=None, order: str='asc', limit: int=100, created_at_gt: str=None, created_at_lte: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity users"
    pagination_token: A token, obtained from previous page. Prior over other filters
        sort: Allowed sort fields
        order: Order by
        limit: Max is 100
        
    """
    url = f"https://monite2.p.rapidapi.com/entity_users"
    querystring = {}
    if first_name:
        querystring['first_name'] = first_name
    if created_at:
        querystring['created_at'] = created_at
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if sort:
        querystring['sort'] = sort
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if order:
        querystring['order'] = order
    if limit:
        querystring['limit'] = limit
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_payable_by_id_payables_payable_id_get(payable_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information about a specific payable with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/payables/{payable_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pending_actions_payables_pending_actions_get(action_type: str='{
  "0": "p",
  "1": "a",
  "2": "y"
}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get next best actions"
    action_type: Action for pending invoice.
        
    """
    url = f"https://monite2.p.rapidapi.com/payables/pending_actions"
    querystring = {}
    if action_type:
        querystring['action_type'] = action_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_user_userpic_entity_users_entity_user_id_userpic_get(entity_user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity user userpic"
    
    """
    url = f"https://monite2.p.rapidapi.com/entity_users/{entity_user_id}/userpic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_by_id_entities_entity_id_get(entity_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information about a specific entity with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/entities/{entity_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_logo_entities_entity_id_logo_get(entity_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an entity logo image"
    
    """
    url = f"https://monite2.p.rapidapi.com/entities/{entity_id}/logo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_user_by_id_entity_users_entity_user_id_get(entity_user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity user by id"
    
    """
    url = f"https://monite2.p.rapidapi.com/entity_users/{entity_user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_payables_payables_get(due_date_lt: str=None, order: str='asc', due_date: str=None, created_at: str=None, created_at_lte: str=None, created_at_gte: str=None, amount_gt: int=None, counterpart_name: str=None, amount_gte: int=None, currency: str='USD', due_date_gt: str=None, status: str='new', due_date_gte: str=None, amount_lt: int=None, created_at_gt: str=None, due_date_lte: str=None, created_at_lt: str=None, amount: int=None, pagination_token: str=None, limit: int=100, amount_lte: int=None, sort: str='{
  "0": "i",
  "1": "d"
}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all payables from the connected entity."
    order: Order by
        pagination_token: A token, obtained from previous page. Prior over other filters
        limit: Max is 100
        sort: Allowed sort fields
        
    """
    url = f"https://monite2.p.rapidapi.com/payables"
    querystring = {}
    if due_date_lt:
        querystring['due_date__lt'] = due_date_lt
    if order:
        querystring['order'] = order
    if due_date:
        querystring['due_date'] = due_date
    if created_at:
        querystring['created_at'] = created_at
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if amount_gt:
        querystring['amount__gt'] = amount_gt
    if counterpart_name:
        querystring['counterpart_name'] = counterpart_name
    if amount_gte:
        querystring['amount__gte'] = amount_gte
    if currency:
        querystring['currency'] = currency
    if due_date_gt:
        querystring['due_date__gt'] = due_date_gt
    if status:
        querystring['status'] = status
    if due_date_gte:
        querystring['due_date__gte'] = due_date_gte
    if amount_lt:
        querystring['amount__lt'] = amount_lt
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if due_date_lte:
        querystring['due_date__lte'] = due_date_lte
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if amount:
        querystring['amount'] = amount
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if limit:
        querystring['limit'] = limit
    if amount_lte:
        querystring['amount__lte'] = amount_lte
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entities_entities_get(created_at_gt: str=None, created_at: str=None, created_at_gte: str=None, pagination_token: str=None, limit: int=100, created_at_lt: str=None, created_at_lte: str=None, internal_id: str=None, name: str=None, type: str='individual', sort: str='{
  "0": "u",
  "1": "p",
  "2": "d",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', order: str='asc', country_code: str='AF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all entities managed by the partner currently logged into Monite."
    pagination_token: A token, obtained from previous page. Prior over other filters
        limit: Max is 100
        sort: Allowed sort fields
        order: Order by
        
    """
    url = f"https://monite2.p.rapidapi.com/entities"
    querystring = {}
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at:
        querystring['created_at'] = created_at
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if limit:
        querystring['limit'] = limit
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if internal_id:
        querystring['internal_id'] = internal_id
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if sort:
        querystring['sort'] = sort
    if order:
        querystring['order'] = order
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_auth_status_entities_entity_id_access_get(entity_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check that you are logged in to Monite as an Entity.
		    A 200 OK response indicates that the current user is still logged in."
    
    """
    url = f"https://monite2.p.rapidapi.com/entities/{entity_id}/access"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag_tags_tag_id_get(tag_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a tag with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/tags/{tag_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments_comments_object_type_object_id_get(object_id: str, object_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments"
    
    """
    url = f"https://monite2.p.rapidapi.com/comments/{object_type}/{object_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_personal_info_info_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read personal info"
    
    """
    url = f"https://monite2.p.rapidapi.com/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supported_export_formats_data_exports_supported_formats_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://monite2.p.rapidapi.com/data_exports/supported_formats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterpart_contacts_counterparts_counterpart_id_contacts_get(counterpart_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get counterpart contacts"
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts/{counterpart_id}/contacts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_auth_status_auth_entity_user_id_get(entity_user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check that the current user is logged in to Monite as an Entity Users.
		    A 200 OK response indicates that the current user is still logged in."
    
    """
    url = f"https://monite2.p.rapidapi.com/auth/{entity_user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_roles_roles_get(created_at_lt: str=None, created_at_gte: str=None, sort: str='{
  "0": "u",
  "1": "p",
  "2": "d",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', pagination_token: str=None, created_at_lte: str=None, name: str=None, created_at: str=None, limit: int=100, created_at_gt: str=None, order: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of roles"
    sort: Allowed sort fields
        pagination_token: A token, obtained from previous page. Prior over other filters
        limit: Max is 100
        order: Order by
        
    """
    url = f"https://monite2.p.rapidapi.com/roles"
    querystring = {}
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if sort:
        querystring['sort'] = sort
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if name:
        querystring['name'] = name
    if created_at:
        querystring['created_at'] = created_at
    if limit:
        querystring['limit'] = limit
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterpart_contact_by_id_counterparts_counterpart_id_contacts_contact_id_get(counterpart_id: str, contact_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get counterpart contact"
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts/{counterpart_id}/contacts/{contact_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def me_me_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about the user currently logged into Monite.
		    The return parameters include historical and activity information about this user."
    
    """
    url = f"https://monite2.p.rapidapi.com/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterpart_bank_accounts_counterparts_counterpart_id_bank_accounts_get(counterpart_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all bank accounts associated with the specified counterpart."
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts/{counterpart_id}/bank_accounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_settings_settings_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity settings."
    
    """
    url = f"https://monite2.p.rapidapi.com/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vat_classes_vat_classes_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all VAT classes sorted by creation date, with the most recently created invoices appearing first."
    
    """
    url = f"https://monite2.p.rapidapi.com/vat_classes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_role_by_id_roles_role_id_get(role_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get role by ID"
    
    """
    url = f"https://monite2.p.rapidapi.com/roles/{role_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_user_userpic_info_userpic_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity user userpic"
    
    """
    url = f"https://monite2.p.rapidapi.com/info/userpic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comment_by_id_comments_comment_id_get(comment_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comment"
    
    """
    url = f"https://monite2.p.rapidapi.com/comments/{comment_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterparts_counterparts_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all counterparts sorted by creation date, with the most recently created counterparts appearing first."
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_products_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the good, material, or service with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/products"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_todo_tasks_assigned_todo_tasks_get(include_related_object_information: bool, status: str='active', limit: int=100, object_id: str=None, object_type: str='partner', sort: str='{
  "0": "c",
  "1": "r",
  "2": "e",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', order: str='asc', pagination_token: str=None, object_id_isnull: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all assigned todo tasks"
    limit: Max is 100
        sort: Allowed sort fields
        order: Order by
        pagination_token: A token, obtained from previous page. Prior over other filters
        
    """
    url = f"https://monite2.p.rapidapi.com/todo_tasks"
    querystring = {'include_related_object_information': include_related_object_information, }
    if status:
        querystring['status'] = status
    if limit:
        querystring['limit'] = limit
    if object_id:
        querystring['object_id'] = object_id
    if object_type:
        querystring['object_type'] = object_type
    if sort:
        querystring['sort'] = sort
    if order:
        querystring['order'] = order
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if object_id_isnull:
        querystring['object_id__isnull'] = object_id_isnull
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterpart_by_id_counterparts_counterpart_id_get(counterpart_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the counterpart with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts/{counterpart_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_auth_status_auth_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check that you are logged in to Monite as an API Partner.
		    A 200 OK response indicates that the current user is still logged in."
    
    """
    url = f"https://monite2.p.rapidapi.com/auth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subject_body_template_mail_contents_get(document_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subject-Body template for Receivables"
    document_type: Document types for receivables
        
    """
    url = f"https://monite2.p.rapidapi.com/mail_contents"
    querystring = {'document_type': document_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tags_tags_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all tags that can be assigned to payables.
		    Tags can be used, for example, as trigger conditions in payable approval workflows."
    
    """
    url = f"https://monite2.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_audit_trail_data_audit_logs_get(created_at_gte: str=None, created_at_gt: str=None, created_at_lt: str=None, created_at: str=None, order: str='asc', target_resource: str=None, sort: str='{
  "0": "c",
  "1": "r",
  "2": "e",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', limit: int=100, pagination_token: str=None, created_at_lte: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get audit trail"
    order: Order by
        sort: Allowed sort fields
        limit: Max is 100
        pagination_token: A token, obtained from previous page. Prior over other filters
        
    """
    url = f"https://monite2.p.rapidapi.com/audit_logs"
    querystring = {}
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if created_at:
        querystring['created_at'] = created_at
    if order:
        querystring['order'] = order
    if target_resource:
        querystring['target_resource'] = target_resource
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_receivable_by_id_receivables_receivable_id_get(receivable_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the invoice with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/receivables/{receivable_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_permissions_permissions_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available permissions"
    
    """
    url = f"https://monite2.p.rapidapi.com/permissions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_receivable_pdf_link_receivables_receivable_id_pdf_link_get(receivable_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves link to generated PDF of a receivable with the given ID"
    
    """
    url = f"https://monite2.p.rapidapi.com/receivables/{receivable_id}/pdf_link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vat_class_by_id_vat_classes_vat_class_id_get(vat_class_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all VAT classes sorted by creation date, with the most recently created invoices appearing first."
    
    """
    url = f"https://monite2.p.rapidapi.com/vat_classes/{vat_class_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_data_exports_get(limit: int=100, pagination_token: str=None, created_by_entity_user_id: str=None, order: str='asc', sort: str='{
  "0": "c",
  "1": "r",
  "2": "e",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: Max is 100
        pagination_token: A token, obtained from previous page. Prior over other filters
        order: Order by
        sort: Allowed sort fields
        
    """
    url = f"https://monite2.p.rapidapi.com/data_exports"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if created_by_entity_user_id:
        querystring['created_by_entity_user_id'] = created_by_entity_user_id
    if order:
        querystring['order'] = order
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_by_id_products_product_id_get(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all goods, materials, or services paginated."
    
    """
    url = f"https://monite2.p.rapidapi.com/products/{product_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_payment_terms_by_id_payment_terms_payment_terms_id_get(payment_terms_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a payment term by its ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/payment_terms/{payment_terms_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mail_templates_template_id_get(template_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get custom template by ID"
    template_id: Integer template ID
        
    """
    url = f"https://monite2.p.rapidapi.com/mail_templates/{template_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_receivables_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all invoices sorted by creation date, with the most recently created invoices appearing first."
    
    """
    url = f"https://monite2.p.rapidapi.com/receivables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_variables_mail_contents_variables_get(receivable_type: str, counterpart_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of placeholders allowed to insert into an email template for customization"
    
    """
    url = f"https://monite2.p.rapidapi.com/mail_contents/variables"
    querystring = {'receivable_type': receivable_type, 'counterpart_type': counterpart_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_one_todo_task_todo_tasks_todo_task_id_get(todo_task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read one todo task"
    
    """
    url = f"https://monite2.p.rapidapi.com/todo_tasks/{todo_task_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_payment_terms_payment_terms_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all configured payment terms."
    
    """
    url = f"https://monite2.p.rapidapi.com/payment_terms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_mail_templates_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all custom templates"
    
    """
    url = f"https://monite2.p.rapidapi.com/mail_templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mailboxes_mailboxes_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the list of mailboxes associated with this Entity."
    
    """
    url = f"https://monite2.p.rapidapi.com/mailboxes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_counterpart_bank_account_by_id_counterparts_counterpart_id_bank_accounts_bank_account_id_get(counterpart_id: str, bank_account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the details for one of the counterpart's bank accounts that is specified by ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/counterparts/{counterpart_id}/bank_accounts/{bank_account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_workflows_workflows_get(created_at_gte: str=None, created_at_lte: str=None, created_at_lt: str=None, created_by: str=None, created_at_gt: str=None, created_at: str=None, sort: str='{
  "0": "c",
  "1": "r",
  "2": "e",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', object_type: str='partner', order: str='asc', limit: int=100, pagination_token: str=None, policy_name: str=None, status: str='active', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get workflows"
    sort: Allowed sort fields
        order: Order by
        limit: Max is 100
        pagination_token: A token, obtained from previous page. Prior over other filters
        
    """
    url = f"https://monite2.p.rapidapi.com/workflows"
    querystring = {}
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if created_by:
        querystring['created_by'] = created_by
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at:
        querystring['created_at'] = created_at
    if sort:
        querystring['sort'] = sort
    if object_type:
        querystring['object_type'] = object_type
    if order:
        querystring['order'] = order
    if limit:
        querystring['limit'] = limit
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if policy_name:
        querystring['policy_name'] = policy_name
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def how_many_tasks_do_i_have_todo_tasks_counters_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tasks count"
    
    """
    url = f"https://monite2.p.rapidapi.com/todo_tasks/counters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_bank_accounts_bank_accounts_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity bank accounts"
    
    """
    url = f"https://monite2.p.rapidapi.com/bank_accounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_system_templates_mail_templates_system_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all system templates"
    
    """
    url = f"https://monite2.p.rapidapi.com/mail_templates/system"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_document_export_by_id_data_exports_id_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://monite2.p.rapidapi.com/data_exports/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_all_approvals_get(order: str='asc', limit: int=100, created_at: str=None, sort: str='{
  "0": "u",
  "1": "p",
  "2": "d",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', created_at_gte: str=None, created_at_gt: str=None, created_at_lt: str=None, created_at_lte: str=None, object_type: str='partner', status: str='active', pagination_token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get approvals"
    order: Order by
        limit: Max is 100
        sort: Allowed sort fields
        pagination_token: A token, obtained from previous page. Prior over other filters
        
    """
    url = f"https://monite2.p.rapidapi.com/approvals"
    querystring = {}
    if order:
        querystring['order'] = order
    if limit:
        querystring['limit'] = limit
    if created_at:
        querystring['created_at'] = created_at
    if sort:
        querystring['sort'] = sort
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if object_type:
        querystring['object_type'] = object_type
    if status:
        querystring['status'] = status
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_pipelines_by_workflow_id_workflows_workflow_id_pipelines_get(workflow_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pipelines by workflow ID"
    
    """
    url = f"https://monite2.p.rapidapi.com/workflows/{workflow_id}/pipelines"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_bank_account_bank_accounts_entity_bank_account_id_get(entity_bank_account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity bank account"
    
    """
    url = f"https://monite2.p.rapidapi.com/bank_accounts/{entity_bank_account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_single_workflow_workflows_workflow_id_get(workflow_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get workflow"
    
    """
    url = f"https://monite2.p.rapidapi.com/workflows/{workflow_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domains_mailbox_domains_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get domains"
    
    """
    url = f"https://monite2.p.rapidapi.com/mailbox_domains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_single_pipeline_workflows_pipelines_pipeline_id_get(pipeline_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pipeline"
    
    """
    url = f"https://monite2.p.rapidapi.com/workflows/pipelines/{pipeline_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_by_id_approvals_approval_id_get(approval_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get approval by ID"
    
    """
    url = f"https://monite2.p.rapidapi.com/approvals/{approval_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_pipelines_workflows_pipelines_get(created_at_gte: str=None, limit: int=100, created_at: str=None, created_at_lt: str=None, status: str='in_progress', created_at_lte: str=None, created_at_gt: str=None, pagination_token: str=None, object_type: str='partner', order: str='asc', sort: str='{
  "0": "c",
  "1": "r",
  "2": "e",
  "3": "a",
  "4": "t",
  "5": "e",
  "6": "d",
  "7": "_",
  "8": "a",
  "9": "t"
}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pipelines"
    limit: Max is 100
        pagination_token: A token, obtained from previous page. Prior over other filters
        order: Order by
        sort: Allowed sort fields
        
    """
    url = f"https://monite2.p.rapidapi.com/workflows/pipelines"
    querystring = {}
    if created_at_gte:
        querystring['created_at__gte'] = created_at_gte
    if limit:
        querystring['limit'] = limit
    if created_at:
        querystring['created_at'] = created_at
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if status:
        querystring['status'] = status
    if created_at_lte:
        querystring['created_at__lte'] = created_at_lte
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if object_type:
        querystring['object_type'] = object_type
    if order:
        querystring['order'] = order
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_measure_unit_by_id_measure_units_measure_unit_id_get(measure_unit_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the measuring unit with the given ID."
    
    """
    url = f"https://monite2.p.rapidapi.com/measure_units/{measure_unit_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_measure_units_measure_units_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all measure units sorted by creation date, with the most recently created invoices appearing first."
    
    """
    url = f"https://monite2.p.rapidapi.com/measure_units"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_document_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://monite2.p.rapidapi.com/document_templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def document_template_by_id(document_template_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://monite2.p.rapidapi.com/document_templates/{document_template_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monite2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

