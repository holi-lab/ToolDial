import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_the_published_landing_pages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of published landing pages of the user."
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/landings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_contacts_groups(fax: str="['']", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of existing contacts groups, paginated."
    fax: The Fax number of the contact
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/groups"
    querystring = {}
    if fax:
        querystring['fax'] = fax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_mo_received_messages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This feature is available only for French customers!. Returns a list (limited to max. 100 entries) of MO messages received for the give type which could be STOP or CONTACT or OTHER; type is a mandatory parameter, the other are optional."
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/mosmshistory"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_received_sms_messages_after_a_specified_message(id_sim: str, id_message: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list (limited to max. 100 entries) of SMS messages received after the given SMS message ID id_message, for the given SIM id_sim."
    id_sim: The phone number where you enabled the service of reception, in international format (+393471234567 or 00393471234567). If no SIM is specified, returns the new messages for all enabled SIMs. It is possible to specify more numbers, separated by comma	
        id_message: The reference message ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/srsmshistory/{id_sim}/{id_message}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_received_sms_messages_for_one_or_more_specific_sims(id_sim: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the paginated list of received SMS messages."
    id_sim: The phone number where you enabled the service of reception, in international format (+393471234567 or 00393471234567). If no SIM is specified, returns the new messages for all enabled SIMs. It is possible to specify more numbers, separated by comma	
        
    """
    url = f"https://send-sms4.p.rapidapi.comAPI/v1.0/REST/srsmshistory/{id_sim}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_received_messages_for_all_sims(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of received messages on the given sim_id since the last call of this method, or the beginning of time if it’s the first call. The list is limited to max. 100 entries.   "
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/newsrsmsmessage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_campaigns_by_statuses(company_url: str, company_fax: str, company_phone: str, email_sender: str, company_name: str, title: str, id_language: str, reply_to: str, company_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of campaigns by statuses. If statuses is not specified by default return only ACTIVE campaign."
    company_url: The company website URL	
        company_fax: The company FAX number	
        company_phone: The company phone number	
        email_sender: The sender’s email address	
        company_name: The company name	
        title: The list name
        id_language: The list default language	
        reply_to: The email to be used for the ‘reply_to’ email field	
        company_address: The company address	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/list"
    querystring = {'company_url	': company_url, 'company_fax	': company_fax, 'company_phone': company_phone, 'email_sender	': email_sender, 'company_name	': company_name, 'title': title, 'id_language': id_language, 'reply_to	': reply_to, 'company_address': company_address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_status(getmoney: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to retrieve the services status of the user identified by the id."
    getmoney: Add user current money to response
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/status"
    querystring = {}
    if getmoney:
        querystring['getMoney'] = getmoney
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_contact(contactid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a contact’s details"
    contactid: The contact ID
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/contact/{contactid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_contact_s_custom_fields(address: str='""', surname: str='""', fax: str='""', phonenumber: str='""', city: str='""', zip: str='""', birthdate: str='""', gender: str='""', groupids: str='[]', province: str='""', name: str='""', sendoptin: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of user’s defined custom fields"
    address: The address of the contact
        surname: The last name of the contact	
        fax: The Fax number of the contact
        phonenumber: The phone number of the contact
        city: The city of the contact
        zip: The ZIP code of the contact
        birthdate: The birth date of the contact  [ddMMyy, yyyyMMdd, ddMMyyHHmm, yyyyMMddHHmmss, yyyy-MM-dd HH:mm, yyyy-MM-dd HH:mm.0]
        gender: The gender of the contact
        groupids: The groups (Ids) in which the contact will be added
        province: The province of the contact	
        name: The first name of the contact	
        sendoptin: Send an Opt-In email to the contact’s email
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/contacts/fields"
    querystring = {}
    if address:
        querystring['address'] = address
    if surname:
        querystring['surname'] = surname
    if fax:
        querystring['fax'] = fax
    if phonenumber:
        querystring['phoneNumber'] = phonenumber
    if city:
        querystring['city'] = city
    if zip:
        querystring['zip'] = zip
    if birthdate:
        querystring['birthdate'] = birthdate
    if gender:
        querystring['gender'] = gender
    if groupids:
        querystring['groupIds'] = groupids
    if province:
        querystring['province'] = province
    if name:
        querystring['name'] = name
    if sendoptin:
        querystring['SendOptIn'] = sendoptin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sent_sms_messages_history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the user’s SMS messages history"
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/smshistory"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_list_of_subaccounts(passwd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of subaccounts of the authenticated user."
    passwd: The subaccount password (min 8 chars, max 16 chars)	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/subaccounts"
    querystring = {'passwd': passwd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_subaccount_s_purchases(passwd: str, subaccount_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the paginated list of purchases made by a subaccount. Caller must be a superaccount."
    passwd: The subaccount password (min 8 chars, max 16 chars)	
        subaccount_id: The subaccount ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/subaccount/{subaccount_id}/purchase"
    querystring = {'passwd': passwd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API used to retrieve the dashboard URL of the authenticated user"
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/dashboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_session(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether the user session is still active and valid (without renewal)"
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/checksession"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authenticate_using_a_user_token(password: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The login with token API lets you authenticate by using your username and password, and returns a token to be used for authenticating the next API calls"
    password: Your account’s dispositive password
        username: Your account’s username
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/token"
    querystring = {'password': password, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authenticate_using_a_session_key(username: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The login with session key API lets you authenticate by using your username and password, and returns a token to be used for authenticating the next API calls"
    username: Your account’s username
        password: Your account’s dispositive password
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/login"
    querystring = {'username': username, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_received_sms_messages_for_all_sims(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the paginated list of received SMS messages."
    
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/srsmshistory"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_aliases(company_name: str, contact_city: str, contact_name: str, contact_info: str, cod_fiscale: str, contact_address: str, vat_number: str, contact_pcode: str, alias: str, contact_surname: str, contact_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a paginated list of the user’s TPOA aliases."
    company_name: The name of company associated to this alias	
        contact_city: The city of company	
        contact_name: The name of physical person rappresenting the company	
        contact_info: The value of contact	
        cod_fiscale: The fiscal code of company	
        contact_address: The address of company	
        vat_number: The vat number of company	
        contact_pcode: The ZIP/Postal-code/CAP code of company	
        alias: The sender that will be shown on SMS	
        contact_surname: The surname of physical person rappresenting the company	
        contact_type: The type of contact referred to contact-info	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/alias"
    querystring = {'company-name	': company_name, 'contact-city	': contact_city, 'contact-name	': contact_name, 'contact-info	': contact_info, 'cod-fiscale	': cod_fiscale, 'contact-address	': contact_address, 'vat-number	': vat_number, 'contact-pcode	': contact_pcode, 'alias': alias, 'contact-surname	': contact_surname, 'contact-type	': contact_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subaccount_available_credits(passwd: str, subaccount_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the available credits of the given subaccount"
    passwd: The subaccount password (min 8 chars, max 16 chars)	
        subaccount_id: The subaccount ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/subaccount/{subaccount_id}/credit"
    querystring = {'passwd': passwd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subaccount(passwd: str, subaccount_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of subaccounts of the authenticated user."
    passwd: The subaccount password (min 8 chars, max 16 chars)	
        subaccount_id: The subaccount ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/subaccount/{subaccount_id}"
    querystring = {'passwd': passwd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_received_messages_for_one_or_more_specific_sims(id_sim: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of received messages on the given sim_id since the last call of this method, or the beginning of time if it’s the first call. The list is limited to max. 100 entries.   "
    id_sim: The phone number where you enabled the service of reception, in international format (+393471234567 or 00393471234567). If no SIM is specified, returns the new messages for all enabled SIMs. It is possible to specify more numbers, separated by comma	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/newsrsmsmessage/{id_sim}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_list_of_phone_numbers_in_blacklist(is_from: str=None, pagesize: str=None, to: str=None, origins: str=None, number: str=None, pagenumber: str=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the paginated list of Phone numbers in SMS Blacklist"
    is_from: Return results from the given date	
        pagesize: The number of results in a page
        to: Return results up to the given date	
        origins: List of origins(API, WEB, STOPSMS) separated by ‘,’ used for filter the results	
        number: Return results that contains this phone number or part of it.	
        pagenumber: Return the given results page	
        timezone: Optional timezone applied to from or to dates	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/blacklist/sms"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if pagesize:
        querystring['pageSize'] = pagesize
    if to:
        querystring['to'] = to
    if origins:
        querystring['origins'] = origins
    if number:
        querystring['number'] = number
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sms_message_state(order_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get informations on the SMS delivery status of the given order_id."
    order_id: The order ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/sms/{order_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_alias(alias_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the data of the given TPOA Alias ID."
    alias_id: The alias ID	
        
    """
    url = f"https://send-sms4.p.rapidapi.com/API/v1.0/REST/alias/{alias_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "send-sms4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

