import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def frontend_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current frontend version
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/frontend/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def network_management_remotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting remote management servers configuration.
		Only one can be active.
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/network/management/remotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logout(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Logs out current logged in user session
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/logout"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def network_timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current timezone
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/network/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about device model, framework version, device name, serial number, firmware version, type of device, current hubrid mode status and api version
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/api/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_video(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return video resolution
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/video"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_volume(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return speaker volume settings
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/volume"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_unlock_card_master(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Master card
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/unlock/card/master"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def desktop_messages_items(limit: int, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all popup messages
		This feature is available only in AA-14FB panel
		"
    limit: Maximum messages on page
        page_number: Page of the messages

        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/desktop/messages/items"
    querystring = {'limit': limit, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_facerecognition(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Face recognition settings
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/facerecognition"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_mode_current(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current panel work mode.
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/mode/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return dashboard information
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/dashboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def network_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns network settings. If DHCP option is enabled,
		response will contain settings gained from DHCP server.
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/network/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Device Time"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_open_remote_denied(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "access denied" via HTTP (for Remote Access Server needs)
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/open/remote/denied"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login(password: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to collect a Token for a registered User
		"
    password: MD5 hash of the password for login (e. g.  123456=E10ADC3949BA59ABBE56E057F20F883E)
        username: The user name for login
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/login"
    querystring = {'password': password, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def desktop_wallpaper(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting path to desktop image
		This feature is disabled in BI-panels
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/desktop/wallpaper"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_open_remote_control_denied(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "access denied" via HTTP (for Remote Access Server needs). In logs will be described as "Access denied by Remote access server"
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/open/remote/control/denied"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_freeaccess(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Create Free access Time Profile
		- Provides free access on proper schedule when pressed 'trade button' or 'info's double tap.
		- This feature is disabled in BI-panels
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/freeaccess"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_led(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return LED auto mode
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/led"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_timeprofile_items_list(sort_field: str=None, filter: str='{
  "filter_field": "apartment_name",
  "filter_type": "equal",
  "filter_format": "string",
  "filter_value": "Baker Street"
}', limit: int=20, page_number: int=2, sort_type: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Time Profile's list"
    sort_field: Field name (JSON Key in item entity)
        filter: Filtering options
        limit: Number of items. Value by default is 10

        page_number: Page number

        sort_type: Sort type (asc/desc)
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/timeprofile/items/list"
    querystring = {}
    if sort_field:
        querystring['sort_field'] = sort_field
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if page_number:
        querystring['page_number'] = page_number
    if sort_type:
        querystring['sort_type'] = sort_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_name_extended(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about device name
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/name/extended"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def network_mac(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns device MAC address"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/network/mac"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forward_legacy_items_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns forward table
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/forward/legacy/items/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_sip_enable(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about Enabling/Disabling SIP
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/sip/enable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_call_dial_timeout(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return dial timeout
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/call/dial/timeout"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_unlock_input_code(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General input code
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/unlock/input/code"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sip_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current SIP registration status
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/sip/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_error_text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns error text notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/error/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_call_talk_timeout(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return talk timeout
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/call/talk/timeout"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def desktop_weather(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set weather settings
		This feature is available only in AA-14FB panel
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/desktop/weather"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_call_concierge(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns concierge number
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/call/concierge"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_call_dial_auto(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Auto dial after digital input
		- This feature is disabled in BI-panels
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/call/dial/auto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_firmware_check_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns info about latest firmware on server"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/firmware/check/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def desktop_statusbar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Desktop statusbar text values
		This feature is available only in AA-14FB panel
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/desktop/statusbar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_rtsp(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User name and password used to access rtsp stream of panel's camera
		Example of panel's camera rtsp stream url: 'rtsp://username:userpassword@device_ip_address:port'
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/rtsp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about device name
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/name"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_sip_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns SIP Settings
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/sip/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_delay_locknumber(locknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time until opening
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/delay/{locknumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apartment_items_list(sort_type: str='asc', page_number: int=2, filter: str='{
  "filter_field": "apartment_name",
  "filter_type": "equal",
  "filter_format": "string",
  "filter_value": "Baker Street"
}', limit: int=20, sort_field: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return apartment's list"
    sort_type: Sort type (asc/desc)
        page_number: Page number

        filter: Filtering options
        limit: Number of items. Value by default is 10

        sort_field: Field name (JSON Key in item entity)
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/apartment/items/list"
    querystring = {}
    if sort_type:
        querystring['sort_type'] = sort_type
    if page_number:
        querystring['page_number'] = page_number
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if sort_field:
        querystring['sort_field'] = sort_field
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_ringing_tone_text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ringing tone notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/ringing/tone/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apartment_book(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show apartments phone book when arrow buttons pressed if option enabled
		This feature is disabled in BI-panels
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/apartment/book"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_payload(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Payload codec
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_elevator_floor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Info about floor for elevator module
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/elevator/floor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_remote_control_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remote access server settings. If enabled - panel sends reques to remote server and |
		waits for response 10 s. If server desn't responde during timeout panels takes access control |
		itself.
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/remote/control/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_timeprofile_item_timeprofileuid(timeprofileuid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Time Profile Entity"
    timeprofileuid: Unique Id for time profile
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/timeprofile/item/{timeprofileuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apartment_item_apartmentuid(apartmentuid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Apartment Entity
		"
    apartmentuid: Unique Identifier for Item
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/apartment/item/{apartmentuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_language(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current system language
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/language"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def network_ntp(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns NTP Settings
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/network/ntp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_settings_ir(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return IR hardwarte motion detector status
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/device/settings/ir"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_open_remote_control_accepted_locknumber(locknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Open lock and trigger "access allowed" event via HTTP (for Remote Access Server needs). In logs will be described as "Lock is opened by Remote access server"
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/open/remote/control/accepted/{locknumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_timeout_locknumber(locknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time after opening until locking
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/timeout/{locknumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_announcement(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Announcement settings
		- This feature is disabled on BI-panels
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/announcement"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_security_monitor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Turn on/off security mode
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/security/monitor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_identifiers_item_identifieruid(identifieruid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Identifier Entity"
    identifieruid: Unique Id for physical/input code identifier
        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/identifiers/item/{identifieruid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_general_lock_open_remote_accepted_locknumber(locknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Open lock and trigger "access allowed" event via HTTP. In logs will be described as "Lock is opened by API call"
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/general/lock/open/remote/accepted/{locknumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_rtsp_feed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "RTSP feed against main camera stream"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/rtsp/feed"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_identifiers_items_list(page_number: int=2, sort_field: str=None, sort_type: str='asc', filter: str='{
  "filter_field": "apartment_name",
  "filter_type": "equal",
  "filter_format": "string",
  "filter_value": "Baker Street"
}', limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Identifier's list"
    page_number: Page number

        sort_field: Field name (JSON Key in item entity)
        sort_type: Sort type (asc/desc)
        filter: Filtering options
        limit: Number of items. Value by default is 10

        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/identifiers/items/list"
    querystring = {}
    if page_number:
        querystring['page_number'] = page_number
    if sort_field:
        querystring['sort_field'] = sort_field
    if sort_type:
        querystring['sort_type'] = sort_type
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_ringing_tone_sound(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ringing tone notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/ringing/tone/sound"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def log_list(to: int=1549286120038, is_from: int=1549286120038, limit: int=20, filter: str='{
  "filter_field": "category",
  "filter_type": "equal",
  "filter_value": "access"
}', sort_type: str='asc', page_number: int=2, locale: str='"en"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return log's list"
    to: Unix time

        is_from: Unix time

        limit: Number of items. Value by default is 10

        filter: Filtering options
        sort_type: Sort type (asc/desc)
        page_number: Page number

        locale: Language for localizing logs. Value by default is 'en'

        
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/log/list"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    if sort_type:
        querystring['sort_type'] = sort_type
    if page_number:
        querystring['page_number'] = page_number
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_lock_open_sound(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lock open notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/lock/open/sound"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_button_press_sound(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns buttons press notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/button/press/sound"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_emergency_text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns error text notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/emergency/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_door_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/door/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_error_sound(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns error text notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/error/sound"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def access_door_sensor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/access/door/sensor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_emergency_sound(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns error text notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/emergency/sound"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def log_email(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SMTP email client config that will be used to send logs on email.
		Log events: outgoing_call, device_reboot, sip_registeration_loss
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/log/email"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forward_mode(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forward strategy
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/forward/mode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_reboot_run(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reboot panel
		"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/reboot/run"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_button_press_text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns buttons press notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/button/press/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def log_email_eventlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of events that will trigger sending email"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/log/email/eventlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_firmware_server_custom(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Custom update server"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/firmware/server/custom"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_notification_lock_open_text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lock open notification settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/advanced/notification/lock/open/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_settings_backup_all(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return backup zip file with all settings"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/settings/backup/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_settings_export_tables(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns zip file with identifiers, apartment, timeprofiles and forwards data"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/settings/export/tables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def system_firmware_update_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Updating status"
    
    """
    url = f"https://bas-ip-multiapartment-panels.p.rapidapi.com/system/firmware/update/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bas-ip-multiapartment-panels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

