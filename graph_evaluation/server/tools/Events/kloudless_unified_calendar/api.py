import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpropertiesforoneservice(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/public/services/$SERVICE_ID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_enableeventsendpoint(cursor: str, is_from: int, until: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/events#events-list-events)
		
		In order to use the events endpoints, you will need to enable event collection via your Kloudless application [Events Configuration tab](https://developers.kloudless.com/applications/*/events-details).
		
		Once you have clicked both event collection boxes, hit Save. Your app will now be able to use the Events endpoints.
		
		Please note, events will not be displayed retro-actively. In order to see events using this endpoint, you will need to perform some sort of action on files or folders in the connected account.
		
		The next API request is a file upload. Once you have completed the POST request successfully, try this endpoint and see what is returned!"
    cursor: `string` (optional)
The last cursor in the event stream that your application has seen. The cursor can also be set to `after-auth`, which will retrieve events that have occurred after the account was connected. This is useful if prior events are unnecessary.

This parameter cannot be used with `from` and `until`. Subsequent cursors returned by a request using those parameters will be restricted to the time range specified in the first request.

In addition, the latest `upstream_cursor` can be provided as a cursor to retrieve any events after that point in the event stream.
        from: `string` (optional)
An ISO-8601 string specifying the time of the oldest event in the range to return (inclusive). Can be used without `until`.
        until: `string` (optional)
An ISO-8601 string specifying the latest time prior to which to return events (exclusive). Must be used along with `from`. Non-overlapping queries can be performed since the event times are in the range [`from`, `until`). Future cursor values returned are limited in range to the `until` value provided in the first request.
        page_size: `number` (optional)
Number of events in the stream to return, if available. The `page_size` must be greater than or equal to `1` and less than or equal to `1000`. Treated as advisory.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/events"
    querystring = {'cursor': cursor, 'from': is_from, 'until': until, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_downloadafile_scontents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/files/$FILE_ID/contents/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinfoaboutaspecificaccount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/authentication#accounts-retrieve-an-account)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_searchforyourfile(parents: int, q: int, lang: str, page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#locate-files-search)"
    parents: `string` (optioanl)
Folder IDs to search on, delimited by commas.
        q: `string`
Terms to search cloud storage for.
        lang: `string` (optional)
Type of Search query. `keyword` is the default and will search the service for specific terms, using the service’s capabilities. `raw` will search the service using the service’s own query language. `cmis` is intended for CMIS queries.
        page: `integer` (optional)
The page to return of a paginated list.
        page_size: `integer` (optional)
Number of results to return per page.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/search/"
    querystring = {'parents': parents, 'q': q, 'lang': lang, 'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievetaskstatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The task object has the following attributes:
		
		* `id`: identifier used to reference task in the Task endpoint.
		
		* `status`: Current status of the request, one of:
		  * `PENDING`: The task state is unknown.
		  * `RECIEVED`: The task was recieved, waiting to start.
		  * `STARTED`: The task was started and currently is in progress.
		
		Once the task has completed, the result of the request will be returned instead of the task metadata.
		
		[Link to API reference](https://developers.kloudless.com/docs/v1/core#asynchronous-requests-and-the-task-api)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/me/tasks/$TASK_ID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmetadataaboutafolder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#folders-retrieve-folder-metadata)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/folders/$FOLDER_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalistofitemsinafolder(page_size: int, recursive: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to our API reference](https://developers.kloudless.com/docs/v1/storage#folders-retrieve-folder-contents)"
    page_size: `integer` (optional)
Number of items to return per page.
        recursive: `boolean` (optional).
List all contents within this folder.
        page: `string` (optional)
Which page of items to retrieve.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/folders/$FOLDER_ID/contents/"
    querystring = {'page_size': page_size, 'recursive': recursive, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalistofallaccounts(admin: int, enabled: int, page_size: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response contains the following information:
		
		* `total` Total number of objects
		* `count` Number of objects on this page
		* `page` Page number
		* `objects` List of account objects
		* `type` Will always be `object_list`
		* `api` Will always be `meta`
		
		[Link to API reference](https://developers.kloudless.com/docs/v1/authentication#accounts-list-accounts)"
    admin: boolean (optional) 
Retrieves only admin/non-admin accounts, if present.
        enabled: boolean (optional) 
Retrieves only enabled/disabled accounts, if present.
        page_size: number (optional) Default: 10 
Number of objects in each page. The page_size must be between 1 and 1000.
        page: number (optional) 
Page to return. page_size number of objects will be returned on each page. By default, the first page is returned. The page parameter can be used to request further objects.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/"
    querystring = {'admin': admin, 'enabled': enabled, 'page_size': page_size, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpermissionsforafile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#permissions)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/files/$FILE_ID/permissions/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getathumbnailforagivenfile(image_format: str, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to the API reference](https://developers.kloudless.com/docs/v1/storage#files-download-a-thumbnail-for-a-file)"
    image_format: `string` (optional)
String specifier for thumbnail format 'png' or 'jpeg'.
        size: `string` (optional)
Size, in pixels, defaults to 256.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/files/$FILE_ID/thumbnail/"
    querystring = {'image_format': image_format, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmetadataaboutaspecficfile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to our API reference](https://developers.kloudless.com/docs/v1/storage#files-retrieve-file-metadata)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/me/storage/files/$FILE_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpermissionsforafolder(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#permissions)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/folders/$FOLDER_ID/permissions/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpropertiesforafile(page: int, page_size: int, active: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#links-list-links)"
    page: `number` (optional)
Page to return. `page_size` will set the number of objects returned on each page.
        page_size: `number` (optional)
Number of objects on each page. The `page_size` must be between `1` and `1000`.
        active: `boolean` (optional)
Retieve only active or only inactive links if present.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/files/$FILE_ID/properties"
    querystring = {'page': page, 'page_size': page_size, 'active': active, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkstothefolderattheservicelevel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to our API reference](https://developers.kloudless.com/api-explorer/#!/accounts/storage_folders_links_read)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/folders/$FOLDER_ID/links/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinformationforaspecificlink(active: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#links-retrieve-a-link)"
    active: `boolean` (optional)
Retrieves only an active/inactive link, if present.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/links/$LINK_ID/"
    querystring = {'active': active, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkstothefileattheservicelevel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to our API reference](https://developers.kloudless.com/api-explorer/#!/accounts/storage_files_links_read)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/files/$FILE_ID/links/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpropertiesforallservices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/public/services/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_getalistofitemsintherootfolder(recursive: int, page_size: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The folder ID `root` can be used to list the contents of the root folder.
		
		
		[API Reference](https://developers.kloudless.com/docs/v1/storage#folders-retrieve-folder-contents)"
    recursive: `boolean` (optional).
List all contents within this folder.
        page_size: `integer` (optional)
Number of items to return per page.
        page: `string` (optional)
Which page of items to retrieve.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/folders/root/contents/"
    querystring = {'recursive': recursive, 'page_size': page_size, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getusagequotadataforaserviceaccount(page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/storage#other-storage-quota)"
    page: `integer` (optional)
A page number within the paginated result set.
        page_size: `integer` (optional)
Number of results to return per page.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/quota/"
    querystring = {'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlinkscreatedforthisaccount(page_size: int, active: int, ordering: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to our API reference](https://developers.kloudless.com/docs/v1/storage#links-list-links)"
    page_size: `integer` (optional)
Number of results to return per page.
        active: `boolean` (optional)
        ordering: `string` (optional)
Which field to use when ordering the results.
        page: `integer` (optional)
A page number within the paginated result set.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/storage/links"
    querystring = {'page_size': page_size, 'active': active, 'ordering': ordering, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievelatestcursor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/events#events-retrieve-latest-cursor)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/events/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallcalendars(page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response contains the following information:
		
		* `count` Number of objects on this page
		
		* `page` Page identifier
		
		* `next_page` The value to provide in the request’s `page` query parameter for the next page. This will be `null` if there are no more pages.
		
		* `objects` List of calendar objects"
    page: `string` (optional) 
Page to return. Do not provide a page parameter when retrieving the first page. To retrieve pages after the first page, set page to the value of next_page found in the previous page of data retrieved.
        page_size: `number` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/cal/calendars"
    querystring = {'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveacalendar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information about a calendar."
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/cal/calendars/$CALENDAR_ID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listevents(start: int, page: int, end: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response contains the following information:
		
		* `count` Number of events on this page
		
		* `page` Page identifier
		
		* `next_page` The value to provide in the request’s `page` query parameter for the next page. This will be `null` if there are no more pages.
		
		* `objects` List of event objects
		
		For Google Calendar, `recurrence` will be unavailable when listing events."
    start: `string` (optional) 
ISO 8601 timestamp indicating the start of the range of events to retrieve, by event start time. The default value is the start of Unix Epoch Time (`1970-01-01T00:00:00Z`).
        page: `string` (optional) 
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        end: `string` (optional) 
ISO 8601 timestamp indicating the end of the range of events to retrieve, by event end time. Required by Outlook Calendar. The default value is the current UTC time.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/cal/calendars/$CALENDAR_ID/events"
    querystring = {'start': start, 'page': page, 'end': end, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallaccounts(page_size: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-accounts-list-accounts)"
    page_size: `integer` (optional)
Number of objects in each page. For some services, the `page_size` is not respected. The `page_size` must be between `1` and `1000`.
Default: 100
        page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/accounts"
    querystring = {'page_size': page_size, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallcontacts(account: int, page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-contacts-list-contacts)"
    account: `string` (optional)
Filter by associated `account_id`. Note: this is only supported in `hubspot`
        page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/contacts/"
    querystring = {'account': account, 'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveanaccount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-accounts-retrieve-an-account)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/accounts/$ACCOUNT_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveanevent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retieve details about an event."
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/cal/calendars/$CALENDAR_ID/events/$EVENT_ID"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveacontact(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-contacts-retrieve-a-contact)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/contacts/$CONTACT_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallleads(page_size: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-leads-list-leads)"
    page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/leads/"
    querystring = {'page_size': page_size, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveacampaign(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-campaigns-retrieve-a-campaign)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/campaigns/$CAMPAIGN_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listalltasks(page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-tasks-list-tasks)"
    page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/tasks/"
    querystring = {'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallcampaigns(page_size: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-campaigns-list-campaigns)"
    page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/campaigns/"
    querystring = {'page_size': page_size, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievealead(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-leads-retrieve-a-lead)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/leads/$LEAD_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveanopportunity(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-opportunities-retrieve-an-opportunity)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/opportunities/$OPPORTUNITY_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallopportunites(page_size: int, page: int, account: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-opportunities-list-opportunities)"
    page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        account: `string` (optional)
Filter by associated `account_id`. Note: this is only supported in `hubspot`
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/opportunites/"
    querystring = {'page_size': page_size, 'page': page, 'account': account, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveatask(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-tasks-retrieve-a-task)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/tasks/$TASK_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveanobject(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-objects-retrieve-an-object)"
    
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/objects/$OBJECT_ID/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallobjects(page: int, page_size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Link to API reference](https://developers.kloudless.com/docs/v1/crm#crm-objects-list-objects)"
    page: `string` (optional)
Page to return. Do not provide a `page` parameter when retrieving the first page. To retrieve pages after the first page, set `page` to the value of `next_page` found in the previous page of data retrieved.
        page_size: `integer` (optional)
Number of objects in each page. For some services, the page_size isn’t respected. The `page_size` must be between `1` and `1000`.
Default: 100
        
    """
    url = f"https://kloudless-unified-calendar.p.rapidapi.com/accounts/me/crm/objects/"
    querystring = {'page': page, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kloudless-unified-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

