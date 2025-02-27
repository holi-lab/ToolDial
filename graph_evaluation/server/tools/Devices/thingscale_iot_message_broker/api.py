import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_user_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User service provides the management functions of the user profile."
    
    """
    url = f"https://sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com/user/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_devices(device_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Device service manages the device registry of ThingScale, aggregates information about all of the devices.
		Each device accumulates the time series data to have been created by the channel (see Channel service).
		When the access of time series data, use Stream service(see Stream service)."
    device_id: The ID of the device.
        
    """
    url = f"https://sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com/device/{device_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_events(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Event service is by setting a threshold for the stream of the channel, make the notification to the email or an external system.
		Before Event service use, please make sure that the stream to the channel is stored."
    event_id: The ID of the event.
        
    """
    url = f"https://sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com/event/{event_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_active_sessions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Session service provides showing pub/sub status to ThingScale's built-in message broker.<br>
		Both MQTT and WebSocket clients shown in session.
		<strong>About how to see the session</strong><br>
		If Publish from the device to the message broker is not a persistent session and disconnects the payload every time it is sent out, the session will be terminated in a very short period of time, so it is not possible to catch the session in most cases.<br>
		When a device subscribes to a message broker, it is inevitably a persistent session so it can be caught until the session ends."
    
    """
    url = f"https://sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com/session"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_channels(channel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<table>
		<th>Special Header<th>Description
		<tr>
		<td>X-WITHTAG (boolean) <td>show tags with channel
		</table>
		(NOTE)Tag name is created when you want to put data containing tags in the channel."
    channel_id: The ID of the channel.
        
    """
    url = f"https://sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com/channel/{channel_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sensinics_tmura-thingscale-iot-message-broker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

