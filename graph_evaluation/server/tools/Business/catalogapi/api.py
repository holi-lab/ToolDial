import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_catalog(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, search: str, max_rank: str, per_page: str, sort: str, format: str, name: str=None, category_id: str=None, min_points: str=None, max_points: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches the catalog based on the parameters passed."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket_id of the catalog you are searching. You can find your available sockets by using the list_available_catalogs method.
        search: Search the name, description and model of items.
        max_rank: Do not return items with a rank higher than this value.
        per_page: The number of items to return, per page. Can be from 1 to 50. Defaults to 10.
        sort: The following sort values are supported:  points desc points asc rank asc score desc
        format: rest or restx
        name: Searchs the name of items
        category_id: Returns only items within this category_id.
        min_points: Return only items that have a point value of at least this value.
        max_points: Return only items that have a point value of no more than this value.
        page: The page number. Defaults to 1.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/search_catalog"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'search': search, 'max_rank': max_rank, 'per_page': per_page, 'sort': sort, }
    if name:
        querystring['name'] = name
    if category_id:
        querystring['category_id'] = category_id
    if min_points:
        querystring['min_points'] = min_points
    if max_points:
        querystring['max_points'] = max_points
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_order(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, format: str, cart_version: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method places as order using the address and items in the cart. Once the order is placed, the cart is deleted."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        format: rest or restx
        cart_version: If cart_version is passed, this method will only succeed if the passed version matches the version of the current cart.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_order_place"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, }
    if cart_version:
        querystring['cart_version'] = cart_version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catalog_breakdown(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, format: str, is_flat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Breaks down the catalog into its categories"
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket you want to receive the breakdown for. You can find your available sockets by using the list_available_catalogs method.
        format: rest or restx
        is_flat: By default, this method will return categories in a tree structure, with child categories contained within their parents.  If you set is_flat to "1", the categories will be returned as a flat list instead of a tree.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/catalog_breakdown"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, }
    if is_flat:
        querystring['is_flat'] = is_flat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_item(creds_datetime: str, creds_uuid: str, socket_id: str, creds_checksum: str, external_user_id: str, catalog_item_id: str, format: str, option_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removes items from the cart."
    creds_datetime: UTC iso801
        creds_uuid: GUID
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        creds_checksum: checksum
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        catalog_item_id: The catalog_item_id of the item. This item must already exist in the cart.
        format: rest of restx
        option_id: The option_id of the item, if the item has options. This option_id must match the option_id the item already in the cart.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_remove_item"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'socket_id': socket_id, 'creds_checksum': creds_checksum, 'external_user_id': external_user_id, 'catalog_item_id': catalog_item_id, }
    if option_id:
        querystring['option_id'] = option_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def empty(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removes all items in the cart."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        format: rest or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_empty"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_catalogs(format: str, token: str=None, creds_uuid: str=None, creds_datetime: str=None, creds_checksum: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the Available Catalogs"
    format: rest or restx
        creds_uuid: GUID
        creds_datetime: UTC iso8601 datetime
        creds_checksum: checksum
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/list_available_catalogs"
    querystring = {}
    if token:
        querystring['token'] = token
    if creds_uuid:
        querystring['creds_uuid'] = creds_uuid
    if creds_datetime:
        querystring['creds_datetime'] = creds_datetime
    if creds_checksum:
        querystring['creds_checksum'] = creds_checksum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_list(creds_datetime: str, creds_uuid: str, creds_checksum: str, external_user_id: str, per_page: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns a list of order numbers (the Catalog API order numbers, not external_order_number) that match a given external_user_id."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        external_user_id: This is the external_user_id you passed when you placed the order. This method does not work with orders that do not have an external_user_id set.
        per_page: The number of orders to return. Defaults to 10. Can be increased to a maximum of 50.
        page: The page number of results to return when there are more than per_page results.
        format: rest or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/order_list"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'external_user_id': external_user_id, 'per_page': per_page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_item(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, catalog_item_id: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an image and description of the item"
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        catalog_item_id: The catalog_item_id from the search_catalog method.
        format: rest or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/view_item"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'catalog_item_id': catalog_item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unlock(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Unlocks a cart that has been locked via the cart_validate method."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        format: rest or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/carts_unlock"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view(external_user_id: str, creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current address and items in the cart."
    external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        format: rest or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_view"
    querystring = {'external_user_id': external_user_id, 'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, format: str, locked: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validates the address and items in the cart. You should call this method just before placing an order to make sure that the order will not be rejected."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        format: rest or restx
        locked: Set this to "1" to lock the cart. (Defaults to "0") A locked cart cannot be modified, meaning that items cannot be added or removed, and the address cannot be changed. One use for this would be to lock the cart before processing a credit card transaction in your system. You would then be ensured that the item in the cart could not be changed while the transaction is processing. You can only call cart_view, cart_unlock, or cart_order_place on a locked cart.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_validate"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, }
    if locked:
        querystring['locked'] = locked
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def set_address(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, first_name: str, last_name: str, address_1: str, city: str, state_province: str, postal_code: str, country: str, phone_number: str, format: str, address_2: str=None, address_3: str=None, email: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds a shipping address to the cart"
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        first_name: Max length is 40 characters.
        last_name: Max length is 40 characters.
        address_1: Max length is 75 characters.
        state_province: For US states, this must be the two character abbreviation.
        country: The ISO 3166-1 alpha-2 country code of the country the order will be shipped to.
        phone_number: If set, this must be a valid phone number. This will only be used for order support or to contact the addressee to arrange a delivery time for any items that require a signature.
        format: rest or restx
        address_2: Max length is 60 characters.
        address_3: Max length is 60 characters.
        email: If set, this must be a valid email address. We highly recommend that you provide an email address so that we can contact the addressee if there is a problem with the order.
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_set_address"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, 'first_name': first_name, 'last_name': last_name, 'address_1': address_1, 'city': city, 'state_province': state_province, 'postal_code': postal_code, 'country': country, 'phone_number': phone_number, }
    if address_2:
        querystring['address_2'] = address_2
    if address_3:
        querystring['address_3'] = address_3
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def set_item_quantity(creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, catalog_item_id: str, quantity: str, format: str, option_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds items to the cart. The quantity passed to this call overrides the quantity of a duplicate item."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        catalog_item_id: The catalog_item_id from the search_catalog method.
        quantity: The number of items to add. If this item is already in the cart, this quantity will replace the current quantity.
        format: rest or restx
        option_id: The option_id from the search_catalog method. (This is required for items that have options.)
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_set_item_quantity"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, 'catalog_item_id': catalog_item_id, 'quantity': quantity, }
    if option_id:
        querystring['option_id'] = option_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def track_order(creds_datetime: str, creds_uuid: str, creds_checksum: str, order_number: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tracks an order given an order number."
    creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        order_number: The order number you received after placing an order with order_place or cart_order_place.
        format: rest  or restx
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/order_track"
    querystring = {'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'order_number': order_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def add_item(quantity: str, creds_datetime: str, creds_uuid: str, creds_checksum: str, socket_id: str, external_user_id: str, catalog_item_id: str, format: str, option_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds items to the cart"
    quantity: The number of items to add. If this item is already in the cart, this quantity will be added to the current quantity.
        creds_datetime: UTC iso8601
        creds_uuid: GUID
        creds_checksum: checksum
        socket_id: This is the socket that the item is in. You can find your available sockets by using the list_available_catalogs method.
        external_user_id: This is an id from your system that identifies the user that the cart is for. It can contain alphanumeric characters, dashes and underscores.
        catalog_item_id: The catalog_item_id from the search_catalog method.
        format: rest or restx
        option_id: The option_id from the search_catalog method. (This is required for items that have options.)
        
    """
    url = f"https://kkollstedt-catalogapi.p.rapidapi.com/{format}/cart_add_item"
    querystring = {'quantity': quantity, 'creds_datetime': creds_datetime, 'creds_uuid': creds_uuid, 'creds_checksum': creds_checksum, 'socket_id': socket_id, 'external_user_id': external_user_id, 'catalog_item_id': catalog_item_id, }
    if option_id:
        querystring['option_id'] = option_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kkollstedt-catalogapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

