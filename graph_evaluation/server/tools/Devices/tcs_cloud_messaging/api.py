import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def replies(messageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replies are responses to a message sent using dynamic codes"
    messageid: Defines the message ID for which a delivery reply is requested.
        
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/replies/{messageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delivery_notification(jobtrackingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delivery notifications are provided in response to sending a message using the CMC REST API. A delivery notification provides a message delivery status and the unique message ID associated with each destination in the send request. The message ID is unique in the system, and is required in all future requests to determine the message delivery status."
    
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/notifications/{jobtrackingid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delivery_receipts(messageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delivery receipts allow you to determine the delivery status of a message. CMC reports two delivery receipts where applicable by the destination’s carrier. The first receipt is the confirmation that the message was sent to the cell network. The second is confirmation of delivery to the device, if that device’s carrier supports the functionality."
    messageid: Defines 1 to n message IDs for which a delivery receipt is requested. Comma-delimited.
        
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/receipts/{messageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_group(groupname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the group provisioned in the CMC system"
    
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/groups/{groupname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_contact_s(mdns: str=None, all: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the contact provisioned in the CMC system. Contacts are used as message destination, and contacts can be added to a group for sending group messages."
    mdns: Defines the contact(s) that is being queried. This could be a single number or a comma separated list.
        all: Query parameter to be passed in to retrieve all the contacts. If the 'all' query parameter is specified then mdns should not be used.
        
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/contacts/{mdns}"
    querystring = {}
    if mdns:
        querystring['mdns'] = mdns
    if all:
        querystring['all'] = all
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def program_replies_with_keyword(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requesting of program replies allows the client to retrieve unsolicited Mobile Originated (MO) traffic sent to a CTIA short code or fixed code."
    keyword: Defines the program requested.
        
    """
    url = f"https://sms.p.rapidapi.com/rest/v1/programreplies/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

