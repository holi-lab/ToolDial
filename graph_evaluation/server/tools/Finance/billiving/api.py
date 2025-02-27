import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def account_summary(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dashboard data"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/reports/accountsummary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve account information (i.e. plan type, expiration date etc)"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bank_account(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve bank account information"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/bankaccounts/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_estimate(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single estimate"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/estimates/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve mail settings"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/settings/mail"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_invoice(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an existing invoice"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/invoices/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve account settings"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_client(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single client"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/clients/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_catalog_discounts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve catalog discounts (i.e.: Bronze, Silver & Gold)"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/products/discounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_bank_accounts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of bank accounts"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/bankaccounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve an existing product"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/products/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_document_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve document type collection"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/doctypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_due_days_options(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of supported due days"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/duedays"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of supported countries"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_date_formats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of date formats"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/dateformats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_pay_status_options(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of pay status"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/paystatus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_frequency_options(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of frequency (i.e.: weekly, monthly, yearly etc.)"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/recurfrequency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_payment_methods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of payment methods (i.e.: cash, check, credit-card etc.)"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/paymentmethods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_tax_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of tax settings"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/settings/taxes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of supported currencies"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_status_options(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of statuses (used for all entities, i.e.: clients, invoices etc.)"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of supported languages"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_estimates(organizationname: str=None, is_id: str=None, refno: str=None, datefrom: str=None, dateto: str=None, status: int=None, skip: int=None, top: int=None, orderby: str=None, count: int=None, filetype: str=None, companyid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List estimates"
    is_id: Estimate identifier
        refno: Reference number
        skip: Paging - records to skip
        top: Paging - records to fetch
        orderby: Order result set by field
        count: When Count=1 returns record count
        filetype: May be specified for file download. Applicable values are CSV or XSLX
        companyid: Applicable if managing multiple companies to issue estimates
        
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/estimates"
    querystring = {}
    if organizationname:
        querystring['OrganizationName'] = organizationname
    if is_id:
        querystring['Id'] = is_id
    if refno:
        querystring['RefNo'] = refno
    if datefrom:
        querystring['DateFrom'] = datefrom
    if dateto:
        querystring['DateTo'] = dateto
    if status:
        querystring['Status'] = status
    if skip:
        querystring['Skip'] = skip
    if top:
        querystring['Top'] = top
    if orderby:
        querystring['OrderBy'] = orderby
    if count:
        querystring['Count'] = count
    if filetype:
        querystring['FileType'] = filetype
    if companyid:
        querystring['companyid'] = companyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_template_styles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve template style collection"
    
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/definitions/style/templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_invoices(companyid: int=None, count: int=None, datefrom: str=None, dateto: str=None, filetype: str=None, orderby: str=None, organizationname: str=None, refno: str=None, skip: int=None, status: int=None, top: int=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List invoices"
    companyid: Applicable if managing multiple companies to issue invoices
        count: When Count=1 returns record count
        filetype: May be specified for file download. Applicable values are CSV or XSLX
        orderby: Order result set by field
        skip: Paging - records to skip
        top: Paging - records to fetch
        is_id: Invoice identifier
        
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/invoices"
    querystring = {}
    if companyid:
        querystring['CompanyId'] = companyid
    if count:
        querystring['Count'] = count
    if datefrom:
        querystring['DateFrom'] = datefrom
    if dateto:
        querystring['DateTo'] = dateto
    if filetype:
        querystring['FileType'] = filetype
    if orderby:
        querystring['OrderBy'] = orderby
    if organizationname:
        querystring['OrganizationName'] = organizationname
    if refno:
        querystring['RefNo'] = refno
    if skip:
        querystring['Skip'] = skip
    if status:
        querystring['Status'] = status
    if top:
        querystring['Top'] = top
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_clients(freetext: str=None, statusid: int=None, skip: int=None, top: int=None, orderby: str=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of clients"
    skip: Paging - records to skip
        top: Paging - records to fetch
        orderby: Order result set by the specified field
        count: When Count=1 returns record count
        
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/clients"
    querystring = {}
    if freetext:
        querystring['FreeText'] = freetext
    if statusid:
        querystring['StatusId'] = statusid
    if skip:
        querystring['Skip'] = skip
    if top:
        querystring['Top'] = top
    if orderby:
        querystring['OrderBy'] = orderby
    if count:
        querystring['Count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_products(orderby: str=None, count: int=None, skip: int=None, top: int=None, statusid: int=None, freesearch: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a collection of products"
    orderby: Order result set by specified field
        count: When count equals 1 record count is returned
        skip: Records to skip used for paging
        top: Records to fetch used for paging
        statusid: List of statuses is available under Enumerations
        freesearch: Search ItemId, short descripton, full description & tags fields
        
    """
    url = f"https://billiving-billiving-v1.p.rapidapi.com/products"
    querystring = {}
    if orderby:
        querystring['OrderBy'] = orderby
    if count:
        querystring['Count'] = count
    if skip:
        querystring['skip'] = skip
    if top:
        querystring['top'] = top
    if statusid:
        querystring['statusid'] = statusid
    if freesearch:
        querystring['freesearch'] = freesearch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billiving-billiving-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

