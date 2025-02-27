import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calls(relayer: str=None, phone: str=None, created_on: str=None, duration: str=None, call_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the incoming and outgoing calls for your organization, most recent first."
    relayer: which relayer received or placed this call (int) (filterable: relayer)
        phone: the phone number of the caller or callee depending on the call_type (string) (filterable: phone)
        created_on: when the call was received or sent (datetime) (filterable: before and after)
        duration: the duration of the call in seconds (int, 0 for missed calls)
        call_type: one of the following strings: (filterable: call_type) mt_call - Outgoing Call  mo_call - Incoming Call  mo_miss - Missed Incoming Call
        
    """
    url = f"https://community-textit.p.rapidapi.com/calls.json"
    querystring = {}
    if relayer:
        querystring['relayer'] = relayer
    if phone:
        querystring['phone'] = phone
    if created_on:
        querystring['created_on'] = created_on
    if duration:
        querystring['duration'] = duration
    if call_type:
        querystring['call_type'] = call_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-textit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_sms_messages(relayer: str=None, phone: str=None, direction: str=None, text: str=None, created_on: str=None, sent_on: str=None, delivered_on: str=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the SMS activity for your organization, listing the most recent messages first."
    relayer: the id of the Android relayer that sent or received this message (int) (filterable: relayer)
        phone: the phone number of the sender or receiver, depending on direction (string) (filterable: phone)
        direction: the direction of the SMS, either I for incoming messages or O for outgoing (string) (filterable: direction)
        text: the text of the message received, not this is the logical view, this message may have been received as multiple text messages (string)
        created_on: the datetime when this sms was either received by the relayer or created by TextIt (datetime) (filterable: before and after)
        sent_on: for outgoing messages, the datetime when the relayer sent the message (null if not yet sent or an incoming message) (datetime)
        delivered_on: for outgoing messages, the datetime when the relayer delivered the message (null if not yet sent or an incoming message) (datetime)
        status: the status of this message, a string one of: (filterable: status) Q - Message is queued awaiting to be sent S - Message has been sent by the relayer D - Message was delivered to the recipient H - Incoming message was handled by TextIt F - Message was not sent due to a failure
        
    """
    url = f"https://community-textit.p.rapidapi.com/sms.json"
    querystring = {}
    if relayer:
        querystring['relayer'] = relayer
    if phone:
        querystring['phone'] = phone
    if direction:
        querystring['direction'] = direction
    if text:
        querystring['text'] = text
    if created_on:
        querystring['created_on'] = created_on
    if sent_on:
        querystring['sent_on'] = sent_on
    if delivered_on:
        querystring['delivered_on'] = delivered_on
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-textit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_relayers(relayer: str=None, phone: str=None, name: str=None, country: str=None, last_seen: str=None, power_level: str=None, power_status: str=None, power_source: str=None, network_type: str=None, pending_message_count: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A GET returns the list of Android relayers for your organization, in the order of last activity date. Note that all status information for the device is as of the last time it was seen and can be null before the first sync."
    relayer: the id of the the relayer (long)
        phone: the phone number for the relayer (string) (filterable: phone)
        name: the name of this relayer (string)
        country: which country the sim card for this relayer is registered for (string, two letter country code) (filterable: country)
        last_seen: the datetime when this relayer was last seen by TextIt (datetime) (filterable: before and after)
        power_level: the power level of the device (int)
        power_status: the power status, either STATUS_DISCHARGING or STATUS_CHARGING (string)
        power_source: the source of power as reported by Android (string)
        network_type: the type of network the device is connected to as reported by Android (string)
        pending_message_count: how many messages are assigned to this relayer but not yet sent (int)
        
    """
    url = f"https://community-textit.p.rapidapi.com/relayers.json"
    querystring = {}
    if relayer:
        querystring['relayer'] = relayer
    if phone:
        querystring['phone'] = phone
    if name:
        querystring['name'] = name
    if country:
        querystring['country'] = country
    if last_seen:
        querystring['last_seen'] = last_seen
    if power_level:
        querystring['power_level'] = power_level
    if power_status:
        querystring['power_status'] = power_status
    if power_source:
        querystring['power_source'] = power_source
    if network_type:
        querystring['network_type'] = network_type
    if pending_message_count:
        querystring['pending_message_count'] = pending_message_count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-textit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

