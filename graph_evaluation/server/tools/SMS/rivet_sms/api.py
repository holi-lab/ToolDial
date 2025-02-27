import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_message_list(apikey: str, enddate: str, start: int, fromdate: str, length: int, clientid: str, content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the SMS sent list based on the range, from date and end date. Range can be like 1 to 100, 100 to 200."
    apikey: Use Rivet SMS API API key (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        enddate: Date format must be in yyyy-mm-dd
        start: Starting index value to fetch the campaign detail 
        fromdate: Date format must be in yyyy-mm-dd
        length: Ending index value to fetch the campaign detail.
        clientid: Use Rivet SMS API client ID (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/GetSMS"
    querystring = {'ApiKey': apikey, 'enddate': enddate, 'start': start, 'fromdate': fromdate, 'length': length, 'ClientId': clientid, }
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sent_message_status(clientid: str, apikey: str, messageid: str, content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the sent message status of the message using the message id."
    clientid: Use Rivet SMS API client Id(You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        apikey: Use Rivet SMS API API key (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        messageid: Id of the message for which you need its status.
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/MessageStatus"
    querystring = {'ClientId': clientid, 'ApiKey': apikey, 'MessageId': messageid, }
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sender_id(clientid: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of all Sender ID that are registered for your account to send SMS."
    clientid: You can get the Client ID at https://app.rivet.solutions/ApiDocument/ApiDocs, once your account is created.
        apikey: You can get the API key at https://app.rivet.solutions/ApiDocument/ApiDocs, once your account is created.
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/SenderId"
    querystring = {'ClientId': clientid, 'ApiKey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_credit_check(apikey: str, clientid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the credit balance in your account."
    apikey: You can get the API key at https://app.rivet.solutions/ApiDocument/ApiDocs, once your account is created.
        clientid: You can get the Client ID at https://app.rivet.solutions/ApiDocument/ApiDocs, once your account is created.
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/Balance"
    querystring = {'ApiKey': apikey, 'ClientId': clientid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_sms(senderid: str, mobilenumber_message: str, apikey: str, clientid: str, is_flash: str=None, scheduletime: str=None, is_unicode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send bulk sms  to multiple numbers ."
    senderid: Registered sender ID.
        mobilenumber_message: sample: 9138764782364^message1~91344873637^message2~916846465464^message3
        apikey: Use Rivet SMS API API key (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        clientid: Use Rivet SMS API client Id(You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        is_flash: Is_Flash is true or false for flash message
        scheduletime: Date in yyyy-MM-dd HH:MM (only for schedule message)
        is_unicode: Is_Unicode is true ,if the message is unicode message else  false for non-unicode message
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/SendBulkSMS"
    querystring = {'SenderId': senderid, 'MobileNumber_Message': mobilenumber_message, 'ApiKey': apikey, 'ClientId': clientid, }
    if is_flash:
        querystring['Is_Flash'] = is_flash
    if scheduletime:
        querystring['ScheduleTime'] = scheduletime
    if is_unicode:
        querystring['Is_Unicode'] = is_unicode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sendsms(senderid: str, clientid: str, mobilenumbers: str, message: str, apikey: str, content_type: str='application/json', is_unicode: bool=None, scheduletime: str=None, groupid: str=None, is_flash: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send single messages"
    senderid: Use  Rivet SMS  sender ID that is assigned to you (You can get it from https://app.rivet.solutions/Management/ManageSenderID)
        clientid: Use  Rivet SMS API client ID (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        mobilenumbers: Recipient number –  numbers in format - 971 223344566(please enter mobile number with country code, without preceding 00 or +). Multiple recipients can be specified separated by commas.
        message: Write the content of your SMS 
        apikey: Use  Rivet SMS API API key (You can get it from https://app.rivet.solutions/ApiDocument/ApiDocs#)
        is_unicode: Boolean value : Is_Unicode is true for unicode message else false.
        scheduletime: Date in yyyy-MM-dd HH:MM (only for schedule message)
        groupid: Valid group-id of current user (only for group message otherwise leave empty string)
        is_flash: Boolean value: True if flash message else false.
        
    """
    url = f"https://rivet-sms.p.rapidapi.com/SendSMS/"
    querystring = {'SenderId': senderid, 'ClientId': clientid, 'MobileNumbers': mobilenumbers, 'Message': message, 'ApiKey': apikey, }
    if content_type:
        querystring['Content-Type'] = content_type
    if is_unicode:
        querystring['Is_Unicode'] = is_unicode
    if scheduletime:
        querystring['ScheduleTime'] = scheduletime
    if groupid:
        querystring['GroupId'] = groupid
    if is_flash:
        querystring['Is_Flash'] = is_flash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rivet-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

