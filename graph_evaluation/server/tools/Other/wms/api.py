import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_restocking_shortcut_data(restockinggroup: int, product: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Stock management] Fetch required data to shortcut the restocking process for Kegs and assortments
		
		When doing restocking, the normal workflow is:
		1. Go on MS page and choose the right restocking group 1. Validate a product move 1. Go on temporary restocking list 1. Find the right group 1. Open the group detail page and find the previous product to see move detail 1. Fill in or validate move detail
		For Kegs and assortments, a user request made is to directly go to the final move detail page when the user validates a MS picking.
		To achieve that, as the process is different (the user does not search to open the right page, but the system has to find it), we created this endpoint to fetch correct data to redirect to."
    restockinggroup: Restocking group id from which to find the temporary location
        product: The product id we are moving
        
    """
    url = f"https://wms13.p.rapidapi.com/wms/restocking/move-data/{restockinggroup}/{product}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_payment_term_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://linear.app/zx-ventures/issue/OPS-1529/[api]-improve-suppliers-endpoints](https://linear.app/zx-ventures/issue/OPS-1529/%5Bapi%5D-improve-suppliers-endpoints"
    
    """
    url = f"https://wms13.p.rapidapi.com/SupplierPaymentTerm"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_id(id_supplier: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get single supplier by id"
    id_supplier: supplier id
        
    """
    url = f"https://wms13.p.rapidapi.com/suppliers/{id_supplier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_suppliers_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all Suppliers."
    
    """
    url = f"https://wms13.p.rapidapi.com/suppliers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_suppliers_comment_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://linear.app/zx-ventures/issue/OPS-1529/[api]-improve-suppliers-endpoints](https://linear.app/zx-ventures/issue/OPS-1529/%5Bapi%5D-improve-suppliers-endpoints"
    
    """
    url = f"https://wms13.p.rapidapi.com/supplier"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mass_print_batch_in_progress_item(is_id: int, address: str=None, customer: str=None, shippingnumber: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch order preparations awaiting manual validation from a given batch"
    id: Id of the batch in labelling in progress to get details from
        id: pass an optional search string for looking up inventory
        address: search an order prep by a shipping address street
        customer: search an order prep by a customer firstname or lastname
        shippingnumber: search an order prep by its shipping number
        
    """
    url = f"https://wms13.p.rapidapi.com/mass_print_order_preparation_batches/{is_id}/labelling_in_progress"
    querystring = {}
    if address:
        querystring['address'] = address
    if customer:
        querystring['customer'] = customer
    if shippingnumber:
        querystring['shippingNumber'] = shippingnumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mass_print_batch_in_progress_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all batches that are printed and awaiting manual validation.
		The batches are in status "Labelling in progress""
    
    """
    url = f"https://wms13.p.rapidapi.com/mass_print_order_preparation_batches/labelling_in_progress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_query_id(types: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Example endpoint with filters params<br>
		https://api-staging.beverage-platform.com/select_options?types[]=organization&types[]=warehouse&types[]=supplier"
    types: organization
        
    """
    url = f"https://wms13.p.rapidapi.com/select_options"
    querystring = {}
    if types:
        querystring['types[]'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_comments(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all Suppliers comments"
    
    """
    url = f"https://wms13.p.rapidapi.com/supplier_comments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_comment(comment_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one supplier comment by id."
    comment_id: id of comment
        
    """
    url = f"https://wms13.p.rapidapi.com/supplier_comments/{comment_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_filtered_comments_id(supplier: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Suppliers comments filter by supplier id."
    supplier: 1
        
    """
    url = f"https://wms13.p.rapidapi.com/supplier_comments/"
    querystring = {}
    if supplier:
        querystring['supplier'] = supplier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_payment_terms_id(id_payment: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "one supplier payment term by id"
    id_payment: payment id
        
    """
    url = f"https://wms13.p.rapidapi.com/supplier_payment_terms/{id_payment}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_id(id_supplier_order: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get products list from the supplier order."
    id_supplier_order: id of supplier order
        
    """
    url = f"https://wms13.p.rapidapi.com/supplier_order_products/{id_supplier_order}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_payment_terms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suppliers payment terms list."
    
    """
    url = f"https://wms13.p.rapidapi.com/supplier_payment_terms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supplier_orders_id(id_supplier_order: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get purchase order details."
    
    """
    url = f"https://wms13.p.rapidapi.com/supplier_orders/{id_supplier_order}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_order_refunds(order_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By default this endpoint retrieve all order refunds except those with a deleted status.
		Some filters are available"
    order_id: ASC|DESC
        
    """
    url = f"https://wms13.p.rapidapi.com/order_refunds"
    querystring = {}
    if order_id:
        querystring['order[id]'] = order_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_purchase_detail_id(id_supplier_order_detail: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get product purchase details"
    id_supplier_order_detail: id of supplier order
        
    """
    url = f"https://wms13.p.rapidapi.com/supplier_order_details/{id_supplier_order_detail}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wms13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

