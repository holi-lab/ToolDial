import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getattachmentmetadata(attachmentid: str, emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metadata such as name and content-type for a given attachment and email."
    attachmentid: ID of attachment
        emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/attachments/{attachmentid}/metadata"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmissedemail(missedemailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List emails that were missed due to plan limits."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/missed-emails/{missedemailid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailattachments(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of attachment metadata such as name and content-type for a given email if present."
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/attachments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadattachment(emailid: str, attachmentid: str, apikey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified attachment for a given email as a stream / array of bytes. You can find attachment ids in email responses endpoint responses. The response type is application/octet-stream."
    emailid: ID of email
        attachmentid: ID of attachment
        apikey: Can pass apiKey in url for this request if you wish to download the file in a browser. Content type will be set to original content type of the attachment file. This is so that browsers can download the file correctly.
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/attachments/{attachmentid}"
    querystring = {}
    if apikey:
        querystring['apiKey'] = apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadattachmentbase64(attachmentid: str, emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified attachment for a given email as a base 64 encoded string. The response type is application/json. This method is similar to the `downloadAttachment` method but allows some clients to get around issues with binary responses."
    attachmentid: ID of attachment
        emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/attachments/{attachmentid}/base64"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemails(inboxid: str, retrytimeout: int=None, unreadonly: bool=None, sort: str='ASC', since: str=None, mincount: int=None, delaytimeout: int=None, before: str=None, size: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List emails that an inbox has received. Only emails that are sent to the inbox's email address will appear in the inbox. It may take several seconds for any email you send to an inbox's email address to appear in the inbox. To make this endpoint wait for a minimum number of emails use the `minCount` parameter. The server will retry the inbox database until the `minCount` is satisfied or the `retryTimeout` is reached"
    inboxid: Id of inbox that emails belongs to
        retrytimeout: Maximum milliseconds to spend retrying inbox database until minCount emails are returned
        sort: Sort the results by received date and direction ASC or DESC
        since: Exclude emails received before this ISO 8601 date time
        mincount: Minimum acceptable email count. Will cause request to hang (and retry) until minCount is satisfied or retryTimeout is reached.
        before: Exclude emails received after this ISO 8601 date time
        size: Alias for limit. Assessed first before assessing any passed limit.
        limit: Limit the result set, ordered by received date time sort direction. Maximum 100. For more listing options see the email controller
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/emails"
    querystring = {}
    if retrytimeout:
        querystring['retryTimeout'] = retrytimeout
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if mincount:
        querystring['minCount'] = mincount
    if delaytimeout:
        querystring['delayTimeout'] = delaytimeout
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrawemailjson(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a raw, unparsed, and unprocessed email wrapped in a JSON response object for easier handling when compared with the getRawEmail text/plain response"
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/raw/json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadbody(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified email body for a given email as a string"
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/body"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailhtmlquery(htmlselector: str, emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse an email body and return the content as an array of text. HTML parsing uses JSoup which supports JQuery/CSS style selectors"
    htmlselector: HTML selector to search for. Uses JQuery/JSoup/CSS style selector like '.my-div' to match content. See https://jsoup.org/apidocs/org/jsoup/select/Selector.html for more information.
        emailid: ID of email to perform HTML query on
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/htmlQuery"
    querystring = {'htmlSelector': htmlselector, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def waitforemailcount(inboxid: str, count: int, delay: int=None, sort: str='ASC', since: str=None, before: str=None, timeout: int=None, unreadonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If inbox contains count or more emails at time of request then return count worth of emails. If not wait until the count is reached and return those or return an error if timeout is exceeded."
    inboxid: Id of the inbox we are fetching emails from
        count: Number of emails to wait for. Must be greater that 1
        delay: Max milliseconds delay between calls
        sort: Sort direction
        since: Filter for emails that were received after the given timestamp
        before: Filter for emails that were received before the given timestamp
        timeout: Max milliseconds to wait
        unreadonly: Optional filter for unread only
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/waitForEmailCount"
    querystring = {'inboxId': inboxid, 'count': count, }
    if delay:
        querystring['delay'] = delay
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if timeout:
        querystring['timeout'] = timeout
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestemail(inboxids: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the newest email in all inboxes or in a passed set of inbox IDs"
    inboxids: Optional set of inboxes to filter by. Only get the latest email from these inbox IDs. If not provided will search across all inboxes
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/latest"
    querystring = {}
    if inboxids:
        querystring['inboxIds'] = inboxids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailtextlines(emailid: str, decodehtmlentities: bool=None, lineseparator: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse an email body and return the content as an array of strings. HTML parsing uses JSoup and UNIX line separators."
    emailid: ID of email to fetch text for
        decodehtmlentities: Decode HTML entities
        lineseparator: Line separator character
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/textLines"
    querystring = {}
    if decodehtmlentities:
        querystring['decodeHtmlEntities'] = decodehtmlentities
    if lineseparator:
        querystring['lineSeparator'] = lineseparator
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailpreviewurls(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of URLs for email content as text/html or raw SMTP message for viewing the message in a browser."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/urls"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getunreademailcount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of emails unread. Unread means has not been viewed in dashboard or returned in an email API response"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/unreadCount"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemaillinks(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HTML parsing uses JSoup and UNIX line separators. Searches content for href attributes"
    emailid: ID of email to fetch text for
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/links"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailhtml(emailid: str, decode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve email content as HTML response for viewing in browsers. Decodes quoted-printable entities and converts charset to UTF-8. Pass your API KEY as a request parameter when viewing in a browser: `?apiKey=xxx`. Returns content-type `text/html;charset=utf-8` so you must call expecting that content response not JSON. For JSON response see the `getEmailHTMLJson` method."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/html"
    querystring = {}
    if decode:
        querystring['decode'] = decode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexpirationdefaults(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return default times used for inbox expiration"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/expired/defaults"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallmissedemails(searchfilter: str=None, page: int=0, since: str=None, sort: str='ASC', before: str=None, size: int=20, inboxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    searchfilter: Optional search filter
        page: Optional page index in list pagination
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        size: Optional page size in list pagination
        inboxid: Optional inbox ID filter
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/missed-emails"
    querystring = {}
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if inboxid:
        querystring['inboxId'] = inboxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadbodybytes(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified email body for a given email as a stream / array of bytes."
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/body-bytes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def waitfornthemail(before: str=None, timeout: int=None, index: int=0, unreadonly: bool=None, inboxid: str=None, sort: str='ASC', since: str=None, delay: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If nth email is already present in inbox then return it. If not hold the connection open until timeout expires or the nth email is received and returned."
    before: Filter for emails that were received before the given timestamp
        timeout: Max milliseconds to wait for the nth email if not already present
        index: Zero based index of the email to wait for. If an inbox has 1 email already and you want to wait for the 2nd email pass index=1
        unreadonly: Optional filter for unread only
        inboxid: Id of the inbox you are fetching emails from
        sort: Sort direction
        since: Filter for emails that were received after the given timestamp
        delay: Max milliseconds delay between calls
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/waitForNthEmail"
    querystring = {}
    if before:
        querystring['before'] = before
    if timeout:
        querystring['timeout'] = timeout
    if index:
        querystring['index'] = index
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if inboxid:
        querystring['inboxId'] = inboxid
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if delay:
        querystring['delay'] = delay
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettemplate(templateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get email template"
    templateid: Template ID
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/templates/{templateid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrawsentemailjson(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a raw, unparsed, and unprocessed sent email wrapped in a JSON response object for easier handling when compared with the getRawSentEmail text/plain response"
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{emailid}/raw/json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxbyemailaddress(emailaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a inbox result by email address"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/byEmailAddress"
    querystring = {'emailAddress': emailaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentdeliverystatuses(before: str=None, since: str=None, sort: str='ASC', size: int=20, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all sent email delivery statuses"
    before: Filter by created at before the given timestamp
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        size: Optional page size in delivery status list pagination
        page: Optional page index in delivery status list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/delivery-status"
    querystring = {}
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxruleset(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get inbox ruleset"
    id: ID of inbox ruleset
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/rulesets/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def waitforlatestemail(inboxid: str=None, since: str=None, unreadonly: bool=None, before: str=None, sort: str='ASC', delay: int=None, timeout: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return either the last received email or wait for an email to arrive and return that. If you need to wait for an email for a non-empty inbox set `unreadOnly=true` or see the other receive methods such as `waitForNthEmail` or `waitForEmailCount`."
    inboxid: Id of the inbox we are fetching emails from
        since: Filter for emails that were received after the given timestamp
        unreadonly: Optional filter for unread only.
        before: Filter for emails that were before after the given timestamp
        sort: Sort direction
        delay: Max milliseconds delay between calls
        timeout: Max milliseconds to wait
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/waitForLatestEmail"
    querystring = {}
    if inboxid:
        querystring['inboxId'] = inboxid
    if since:
        querystring['since'] = since
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if delay:
        querystring['delay'] = delay
    if timeout:
        querystring['timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doesinboxexist(emailaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if inboxes exist by email address. Useful if you are sending emails to mailslurp addresses"
    emailaddress: Email address
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/exists"
    querystring = {'emailAddress': emailaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemail(emailid: str, decode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a email summary object with headers and content. To retrieve the raw unparsed email use the getRawEmail endpoints"
    decode: Decode email body quoted-printable encoding to plain text. SMTP servers often encode text using quoted-printable format (for instance `=D7`). This can be a pain for testing
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}"
    querystring = {}
    if decode:
        querystring['decode'] = decode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallinboxes(sort: str='ASC', page: int=0, before: str=None, domainid: str=None, since: str=None, size: int=20, tag: str=None, favourite: bool=None, search: str=None, inboxtype: str='HTTP_INBOX', teamaccess: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List inboxes in paginated form. The results are available on the `content` property of the returned object. This method allows for page index (zero based), page size (how many results to return), and a sort direction (based on createdAt time). You Can also filter by whether an inbox is favorited or use email address pattern. This method is the recommended way to query inboxes. The alternative `getInboxes` method returns a full list of inboxes but is limited to 100 results."
    sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in list pagination
        before: Optional filter by created before given date time
        domainid: Optional domain ID filter
        since: Optional filter by created after given date time
        size: Optional page size in list pagination
        tag: Optionally filter by tags. Will return inboxes that include given tags
        favourite: Optionally filter results for favourites only
        search: Optionally filter by search words partial matching ID, tags, name, and email address
        inboxtype: Optional filter by inbox type
        teamaccess: DEPRECATED. Optionally filter by team access.
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/paginated"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if before:
        querystring['before'] = before
    if domainid:
        querystring['domainId'] = domainid
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    if tag:
        querystring['tag'] = tag
    if favourite:
        querystring['favourite'] = favourite
    if search:
        querystring['search'] = search
    if inboxtype:
        querystring['inboxType'] = inboxtype
    if teamaccess:
        querystring['teamAccess'] = teamaccess
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexpiredinboxes(before: str=None, size: int=20, sort: str='ASC', page: int=0, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Inboxes created with an expiration date will expire after the given date. An ExpiredInboxRecord is created that records the inboxes old ID and email address. You can still read emails in the inbox (using the inboxes old ID) but the email address associated with the inbox can no longer send or receive emails. Fetch expired inbox records to view the old inboxes properties"
    before: Filter by created at before the given timestamp
        size: Optional page size in inbox sent email list pagination
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in inbox sent email list pagination
        since: Filter by created at after the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/expired"
    querystring = {}
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcontact(contactid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/contacts/{contactid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxemailspaginated(inboxid: str, before: str=None, sort: str='ASC', since: str=None, page: int=0, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of emails in an inbox. Does not hold connections open."
    inboxid: Id of inbox that emails belongs to
        before: Optional filter by received before given date time
        sort: Optional createdAt sort direction ASC or DESC
        since: Optional filter by received after given date time
        page: Optional page index in inbox emails list pagination
        size: Optional page size in inbox emails list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/emails/paginated"
    querystring = {}
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcontacts(since: str=None, size: int=20, before: str=None, sort: str='ASC', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    since: Filter by created at after the given timestamp
        size: Optional page size in list pagination
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/contacts/paginated"
    querystring = {}
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getattachments(since: str=None, before: str=None, size: int=20, sort: str='ASC', page: int=0, filenamefilter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all attachments in paginated response. Each entity contains meta data for the attachment such as `name` and `content-type`. Use the `attachmentId` and the download endpoints to get the file contents."
    since: Filter by created at after the given timestamp
        before: Filter by created at before the given timestamp
        size: Optional page size for list pagination
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index for list pagination
        filenamefilter: Optional file name and content type search filter
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/attachments"
    querystring = {}
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if filenamefilter:
        querystring['fileNameFilter'] = filenamefilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalltrackingpixels(since: str=None, before: str=None, searchfilter: str=None, size: int=20, sort: str='ASC', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List tracking pixels in paginated form"
    since: Filter by created at after the given timestamp
        before: Filter by created at before the given timestamp
        searchfilter: Optional search filter
        size: Optional page size in list pagination
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/tracking/pixels"
    querystring = {}
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if size:
        querystring['size'] = size
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaliasthreads(aliasid: str, page: int=0, size: int=20, before: str=None, since: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns threads created for an email alias in paginated form"
    page: Optional page index in thread list pagination
        size: Optional page size in thread list pagination
        before: Optional filter by sent before given date time
        since: Optional filter by sent after given date time
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/aliases/{aliasid}/threads"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaliasemails(aliasid: str, sort: str='ASC', since: str=None, size: int=20, before: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get paginated emails for an alias by ID"
    sort: Optional createdAt sort direction ASC or DESC
        since: Optional filter by sent after given date time
        size: Optional page size alias email list pagination
        before: Optional filter by sent before given date time
        page: Optional page index alias email list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/aliases/{aliasid}/emails"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestemailininbox_1(inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the newest email in all inboxes or in a passed set of inbox IDs"
    inboxid: ID of the inbox you want to get the latest email from
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/latestIn"
    querystring = {'inboxId': inboxid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettemplatepreviewjson(templateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get email template preview with passed template variables in JSON format. Pass template variables as query params."
    templateid: Template ID
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/templates/{templateid}/preview/json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexpiredinboxrecord(expiredid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Inboxes created with an expiration date will expire after the given date and be moved to an ExpiredInbox entity. You can still read emails in the inbox but it can no longer send or receive emails. Fetch the expired inboxes to view the old inboxes properties"
    expiredid: ID of the ExpiredInboxRecord you want to retrieve. This is different from the ID of the inbox you are interested in. See other methods for getting ExpiredInboxRecord for an inbox inboxId
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/expired/{expiredid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettemplatepreviewhtml(templateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get email template preview with passed template variables in HTML format for browsers. Pass template variables as query params."
    templateid: Template ID
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/templates/{templateid}/preview/html"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listinboxrulesets(inboxid: str, before: str=None, searchfilter: str=None, page: int=0, sort: str='ASC', since: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all rulesets attached to an inbox"
    before: Optional filter by created before given date time
        searchfilter: Optional search filter
        page: Optional page index in inbox ruleset list pagination
        sort: Optional createdAt sort direction ASC or DESC
        since: Optional filter by created after given date time
        size: Optional page size in inbox ruleset list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/rulesets"
    querystring = {}
    if before:
        querystring['before'] = before
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcontactvcard(contactid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/contacts/{contactid}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbouncedemails(page: int=0, size: int=20, since: str=None, sort: str='ASC', before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bounced emails are email you have sent that were rejected by a recipient"
    page: Optional page index
        size: Optional page size 
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/bounce/emails"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexpiredinboxbyinboxid(inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the inboxId to return an ExpiredInboxRecord if an inbox has expired. Inboxes expire and are disabled if an expiration date is set or plan requires. Returns 404 if no expired inbox is found for the inboxId"
    inboxid: ID of inbox you want to retrieve (not the inbox ID)
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/expired/inbox/{inboxid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgroupwithcontactspaginated(groupid: str, sort: str='ASC', since: str=None, page: int=0, before: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get group and paginated contacts belonging to it"
    sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        page: Optional page index in group contact pagination
        before: Filter by created at before the given timestamp
        size: Optional page size in group contact pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/groups/{groupid}/contacts-paginated"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsmtpusername(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/user/smtp/username"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getphonenumber(phonenumberid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/phone/numbers/{phonenumberid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getattachment(attachmentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    attachmentid: ID of attachment
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/attachments/{attachmentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgroups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaliases(size: int=20, since: str=None, sort: str='ASC', before: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all email aliases in paginated form"
    size: Optional page size in alias list pagination
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        page: Optional page index in alias list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/aliases"
    querystring = {}
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcontacts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/contacts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getattachmentinfo(attachmentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metadata for an attachment. It is saved separately to the content of the attachment. Contains properties `name` and `content-type` and `content-length` in bytes for a given attachment."
    attachmentid: ID of attachment
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/attachments/{attachmentid}/metadata"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadattachmentasbase64encoded(attachmentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified attachment for a given email as a base 64 encoded string. The response type is application/json. This method is similar to the `downloadAttachment` method but allows some clients to get around issues with binary responses."
    attachmentid: ID of attachment
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/attachments/{attachmentid}/base64"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloadnewattachment(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/new-attachment-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgroup(groupid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/groups/{groupid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdomain(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns domain verification status and tokens for a given domain"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/domains/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallsenttrackingpixels(before: str=None, size: int=20, since: str=None, sort: str='ASC', page: int=0, searchfilter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all sent email tracking pixels in paginated form"
    before: Filter by created at before the given timestamp
        size: Optional page size in sent email tracking pixel list pagination
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in sent email tracking pixel list pagination
        searchfilter: Optional search filter
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/tracking-pixels"
    querystring = {}
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallgroups(page: int=0, since: str=None, size: int=20, before: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Optional page index in list pagination
        since: Filter by created at after the given timestamp
        size: Optional page size in list pagination
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/groups/paginated"
    querystring = {}
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgroupwithcontacts(groupid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/groups/{groupid}/contacts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemailtrackingpixels(is_id: str, before: str=None, searchfilter: str=None, page: int=0, sort: str='ASC', since: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all tracking pixels for a sent email in paginated form"
    before: Filter by created at before the given timestamp
        searchfilter: Optional search filter
        page: Optional page index in sent email tracking pixel list pagination
        sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        size: Optional page size in sent email tracking pixel list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{is_id}/tracking-pixels"
    querystring = {}
    if before:
        querystring['before'] = before
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadattachmentasbytes(attachmentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the specified attachment for a given email as a stream / array of bytes. You can find attachment ids in email responses endpoint responses. The response type is application/octet-stream."
    attachmentid: ID of attachment
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/attachments/{attachmentid}/bytes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemails(inboxid: str=None, page: int=0, before: str=None, size: int=20, since: str=None, searchfilter: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    inboxid: Optional inboxId to filter sender of sent emails by
        page: Optional page index in inbox sent email list pagination
        before: Filter by created at before the given timestamp
        size: Optional page size in inbox sent email list pagination
        since: Filter by created at after the given timestamp
        searchfilter: Optional search filter
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent"
    querystring = {}
    if inboxid:
        querystring['inboxId'] = inboxid
    if page:
        querystring['page'] = page
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentdeliverystatus(deliveryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a sent email delivery status"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/delivery-status/{deliveryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdomains(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all custom domains you have created"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/domains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemailswithqueueresults(page: int=0, size: int=20, before: str=None, since: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Optional page index in inbox sent email list pagination
        size: Optional page size in inbox sent email list pagination
        before: Filter by created at before the given timestamp
        since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/queue-results"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemailpreviewurls(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of URLs for sent email content as text/html or raw SMTP message for viewing the message in a browser."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{is_id}/urls"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemailhtmlcontent(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{is_id}/html"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentorganizationemails(inboxid: str=None, size: int=20, since: str=None, searchfilter: str=None, before: str=None, page: int=0, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all sent organization emails in paginated form"
    inboxid: Optional inboxId to filter sender of sent emails by
        size: Optional page size in sent email list pagination
        since: Filter by created at after the given timestamp
        searchfilter: Optional search filter
        before: Filter by created at before the given timestamp
        page: Optional page index in sent email list pagination
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/organization"
    querystring = {}
    if inboxid:
        querystring['inboxId'] = inboxid
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if before:
        querystring['before'] = before
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentdeliverystatusesbysentid(sentid: str, before: str=None, sort: str='ASC', page: int=0, since: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all sent email delivery statuses"
    before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in delivery status list pagination
        since: Filter by created at after the given timestamp
        size: Optional page size in delivery status list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{sentid}/delivery-status"
    querystring = {}
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrawsentemailcontents(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a raw, unparsed, and unprocessed sent email. If your client has issues processing the response it is likely due to the response content-type which is text/plain. If you need a JSON response content-type use the getRawSentEmailJson endpoint"
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{emailid}/raw"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsentemail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sent/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxrulesets(inboxid: str=None, searchfilter: str=None, before: str=None, sort: str='ASC', page: int=0, size: int=20, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all rulesets attached to an inbox"
    inboxid: Optional inbox id to get rulesets from
        searchfilter: Optional search filter
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in inbox ruleset list pagination
        size: Optional page size in inbox ruleset list pagination
        since: Filter by created at after the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/rulesets"
    querystring = {}
    if inboxid:
        querystring['inboxId'] = inboxid
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemergencyaddresses(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/phone/emergency-addresses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganizationinboxes(sort: str='ASC', size: int=20, searchfilter: str=None, page: int=0, before: str=None, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List organization inboxes in paginated form. These are inboxes created with `allowTeamAccess` flag enabled. Organization inboxes are `readOnly` for non-admin users. The results are available on the `content` property of the returned object. This method allows for page index (zero based), page size (how many results to return), and a sort direction (based on createdAt time). "
    sort: Optional createdAt sort direction ASC or DESC
        size: Optional page size in list pagination
        searchfilter: Optional search filter
        page: Optional page index in list pagination
        before: Optional filter by created before given date time
        since: Optional filter by created after given date time
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/organization"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxcount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxids(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of inbox IDs"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxsentemails(inboxid: str, sort: str='ASC', before: str=None, size: int=20, since: str=None, searchfilter: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an inbox's sent email receipts. Call individual sent email endpoints for more details. Note for privacy reasons the full body of sent emails is never stored. An MD5 hash hex is available for comparison instead."
    sort: Optional createdAt sort direction ASC or DESC
        before: Optional filter by sent before given date time
        size: Optional page size in inbox sent email list pagination
        since: Optional filter by sent after given date time
        searchfilter: Optional sent email search
        page: Optional page index in inbox sent email list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/sent"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimapsmtpaccess(inboxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IMAP and SMTP access usernames and passwords"
    inboxid: Inbox ID
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/imap-smtp-access"
    querystring = {}
    if inboxid:
        querystring['inboxId'] = inboxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxtags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all inbox tags"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxemailcount(inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    inboxid: Id of inbox that emails belongs to
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/emails/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdeliverystatusesbyinboxid(inboxid: str, page: int=0, sort: str='ASC', before: str=None, size: int=20, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all email delivery statuses for an inbox"
    page: Optional page index in delivery status list pagination
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        size: Optional page size in delivery status list pagination
        since: Filter by created at after the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/delivery-status"
    querystring = {}
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinbox(inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an inbox's properties, including its email address and ID."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestemailininbox(timeoutmillis: int, inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the newest email in an inbox or wait for one to arrive"
    timeoutmillis: Timeout milliseconds to wait for latest email
        inboxid: ID of the inbox you want to get the latest email from
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/getLatestEmail"
    querystring = {'timeoutMillis': timeoutmillis, 'inboxId': inboxid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxbyname(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a inbox result by name"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/byName"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbouncedrecipient(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bounced emails are email you have sent that were rejected by a recipient"
    id: ID of the bounced recipient
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/bounce/recipients/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxes(before: str=None, size: int=100, since: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the inboxes you have created. Note use of the more advanced `getAllEmails` is recommended and allows paginated access using a limit and sort parameter."
    before: Optional filter by created before given date time
        size: Optional result size limit. Note an automatic limit of 100 results is applied. See the paginated `getAllEmails` for larger queries.
        since: Optional filter by created after given date time
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes"
    querystring = {}
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcomplaints(page: int=0, sort: str='ASC', since: str=None, before: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SMTP complaints made against your account"
    page: Optional page index 
        sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        before: Filter by created at before the given timestamp
        size: Optional page size 
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/bounce/complaints"
    querystring = {}
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listinboxtrackingpixels(inboxid: str, sort: str='ASC', searchfilter: str=None, page: int=0, size: int=20, before: str=None, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all tracking pixels sent from an inbox"
    sort: Optional createdAt sort direction ASC or DESC
        searchfilter: Optional search filter
        page: Optional page index in inbox tracking pixel list pagination
        size: Optional page size in inbox tracking pixel list pagination
        before: Optional filter by created before given date time
        since: Optional filter by created after given date time
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/tracking-pixels"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhook(webhookid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/{webhookid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettemplates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all templates"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalltemplates(since: str=None, sort: str='ASC', page: int=0, size: int=20, before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all templates in paginated format"
    since: Filter by created at after the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        page: Optional page index in list pagination
        size: Optional page size in list pagination
        before: Filter by created at before the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/templates/paginated"
    querystring = {}
    if since:
        querystring['since'] = since
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrawemailcontents(emailid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a raw, unparsed, and unprocessed email. If your client has issues processing the response it is likely due to the response content-type which is text/plain. If you need a JSON response content-type use the getRawEmailJson endpoint"
    emailid: ID of email
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/raw"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailhtmljson(emailid: str, decode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve email content as HTML response. Decodes quoted-printable entities and converts charset to UTF-8. Returns content-type `application/json;charset=utf-8` so you must call expecting that content response not JSON."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/{emailid}/html/json"
    querystring = {}
    if decode:
        querystring['decode'] = decode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallunknownmissedemails(since: str=None, size: int=20, page: int=0, sort: str='ASC', inboxid: str=None, before: str=None, searchfilter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Unknown missed emails are emails that were sent to MailSlurp but could not be assigned to an existing inbox."
    since: Filter by created at after the given timestamp
        size: Optional page size in list pagination
        page: Optional page index in list pagination
        sort: Optional createdAt sort direction ASC or DESC
        inboxid: Optional inbox ID filter
        before: Filter by created at before the given timestamp
        searchfilter: Optional search filter
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/missed-emails/unknown"
    querystring = {}
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if inboxid:
        querystring['inboxId'] = inboxid
    if before:
        querystring['before'] = before
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailspaginated(unreadonly: bool=None, searchfilter: str=None, before: str=None, sort: str='ASC', size: int=20, inboxid: str='[]', since: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By default returns all emails across all inboxes sorted by ascending created at date. Responses are paginated. You can restrict results to a list of inbox IDs. You can also filter out read messages"
    unreadonly: Optional filter for unread emails only. All emails are considered unread until they are viewed in the dashboard or requested directly
        searchfilter: Optional search filter. Searches email recipients, sender, subject, email address and ID. Does not search email body
        before: Optional filter emails received before given date time
        sort: Optional createdAt sort direction ASC or DESC
        size: Optional page size in email list pagination. Maximum size is 100. Use page index and sort to page through larger results
        inboxid: Optional inbox ids to filter by. Can be repeated. By default will use all inboxes belonging to your account.
        since: Optional filter emails received after given date time
        page: Optional page index in email list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails"
    querystring = {}
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if inboxid:
        querystring['inboxId'] = inboxid
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgravatarurlforemailaddress(emailaddress: str, size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get gravatar url for email address"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/gravatarFor"
    querystring = {'emailAddress': emailaddress, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemailcount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/emails/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalias(aliasid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an email alias by ID"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/aliases/{aliasid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def waitfornthmissedemail(index: int, timeout: int=None, before: str=None, since: str=None, inboxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Wait for 0 based index missed email"
    index: Zero based index of the email to wait for. If 1 missed email already and you want to wait for the 2nd email pass index=1
        timeout: Optional timeout milliseconds
        before: Filter by created at before the given timestamp
        since: Filter by created at after the given timestamp
        inboxid: Optional inbox ID filter
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/missed-emails/waitForNthMissedEmail"
    querystring = {'index': index, }
    if timeout:
        querystring['timeout'] = timeout
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    if inboxid:
        querystring['inboxId'] = inboxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganizationemailspaginated(page: int=0, searchfilter: str=None, sort: str='ASC', inboxid: str='[]', unreadonly: bool=None, since: str=None, before: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By default returns all emails across all team inboxes sorted by ascending created at date. Responses are paginated. You can restrict results to a list of inbox IDs. You can also filter out read messages"
    page: Optional page index in email list pagination
        searchfilter: Optional search filter search filter for emails.
        sort: Optional createdAt sort direction ASC or DESC
        inboxid: Optional inbox ids to filter by. Can be repeated. By default will use all inboxes belonging to your account.
        unreadonly: Optional filter for unread emails only. All emails are considered unread until they are viewed in the dashboard or requested directly
        since: Optional filter emails received after given date time
        before: Optional filter emails received before given date time
        size: Optional page size in email list pagination. Maximum size is 100. Use page index and sort to page through larger results
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/emails/organization"
    querystring = {}
    if page:
        querystring['page'] = page
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if sort:
        querystring['sort'] = sort
    if inboxid:
        querystring['inboxId'] = inboxid
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloadnewemail(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/new-email-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxforwarder(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get inbox ruleset"
    id: ID of inbox forwarder
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/forwarders/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/user/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsmtppassword(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/user/smtp/password"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvalidationrequests(before: str=None, isvalid: bool=None, size: int=20, searchfilter: str=None, sort: str='DESC', since: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    before: Filter by created at before the given timestamp
        isvalid: Filter where email is valid is true or false
        size: Optional page size for paginated result list.
        searchfilter: Optional search filter
        sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        page: Optional page index in list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/email-verification/validation-requests"
    querystring = {}
    if before:
        querystring['before'] = before
    if isvalid:
        querystring['isValid'] = isvalid
    if size:
        querystring['size'] = size
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxforwarders(searchfilter: str=None, inboxid: str=None, page: int=0, size: int=20, since: str=None, before: str=None, sort: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all forwarders attached to an inbox"
    searchfilter: Optional search filter
        inboxid: Optional inbox id to get forwarders from
        page: Optional page index in inbox forwarder list pagination
        size: Optional page size in inbox forwarder list pagination
        since: Filter by created at after the given timestamp
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/forwarders"
    querystring = {}
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if inboxid:
        querystring['inboxId'] = inboxid
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettrackingpixel(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/tracking/pixels/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloadbouncerecipient(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get webhook test payload for bounce recipient"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/email-bounce-recipient-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallwebhooks(page: int=0, since: str=None, searchfilter: str=None, sort: str='DESC', before: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List webhooks in paginated form. Allows for page index, page size, and sort direction."
    page: Optional page index in list pagination
        since: Filter by created at after the given timestamp
        searchfilter: Optional search filter
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        size: Optional page size for paginated result list.
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/paginated"
    querystring = {}
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemergencyaddress(addressid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/phone/emergency-addresses/{addressid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayload(eventname: str='EMAIL_RECEIVED', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get test webhook payload example. Response content depends on eventName passed. Uses `EMAIL_RECEIVED` as default."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test"
    querystring = {}
    if eventname:
        querystring['eventName'] = eventname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookresults(webhookid: str, unseenonly: bool=None, before: str=None, sort: str='ASC', searchfilter: str=None, size: int=20, since: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    webhookid: ID of webhook to get results for
        unseenonly: Filter for unseen exceptions only
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        searchfilter: Optional search filter
        size: Optional page size in list pagination
        since: Filter by created at after the given timestamp
        page: Optional page index in list pagination
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/{webhookid}/results"
    querystring = {}
    if unseenonly:
        querystring['unseenOnly'] = unseenonly
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinboxwebhookspaginated(inboxid: str, before: str=None, searchfilter: str=None, page: int=0, sort: str='ASC', size: int=20, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    before: Filter by created at before the given timestamp
        searchfilter: Optional search filter
        page: Optional page index in list pagination
        sort: Optional createdAt sort direction ASC or DESC
        size: Optional page size in list pagination
        since: Filter by created at after the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/webhooks/paginated"
    querystring = {}
    if before:
        querystring['before'] = before
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookresult(webhookresultid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    webhookresultid: Webhook Result ID
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/results/{webhookresultid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbouncedemail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bounced emails are email you have sent that were rejected by a recipient"
    id: ID of the bounced email to fetch
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/bounce/emails/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloademailread(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get webhook test payload for email opened event"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/email-read-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbouncedrecipients(size: int=20, before: str=None, sort: str='ASC', since: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bounced recipients are email addresses that you have sent emails to that did not accept the sent email. Once a recipient is bounced you cannot send emails to that address."
    size: Optional page size 
        before: Filter by created at before the given timestamp
        sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        page: Optional page index 
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/bounce/recipients"
    querystring = {}
    if size:
        querystring['size'] = size
    if before:
        querystring['before'] = before
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloadnewcontact(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/new-contact-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhooks(inboxid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/inboxes/{inboxid}/webhooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsmsmessage(smsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a SMS summary object with content."
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sms/{smsid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloadbounce(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get webhook test payload for bounce"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/email-bounce-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookresultsunseenerrorcount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/results/unseen-count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettestwebhookpayloademailopened(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get webhook test payload for email opened event"
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/test/email-opened-payload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsmsmessagespaginated(page: int=0, since: str=None, unreadonly: bool=None, phonenumber: str=None, sort: str='ASC', before: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By default returns all SMS messages across all phone numbers sorted by ascending created at date. Responses are paginated. You can restrict results to a list of phone number IDs. You can also filter out read messages"
    page: Optional page index in SMS list pagination
        since: Optional filter SMSs received after given date time
        unreadonly: Optional filter for unread SMS only. All SMS are considered unread until they are viewed in the dashboard or requested directly
        phonenumber: Optional receiving phone number to filter SMS messages for
        sort: Optional createdAt sort direction ASC or DESC
        before: Optional filter SMSs received before given date time
        size: Optional page size in SMS list pagination. Maximum size is 100. Use page index and sort to page through larger results
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/sms"
    querystring = {}
    if page:
        querystring['page'] = page
    if since:
        querystring['since'] = since
    if unreadonly:
        querystring['unreadOnly'] = unreadonly
    if phonenumber:
        querystring['phoneNumber'] = phonenumber
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getphonenumbers(page: int=0, size: int=20, sort: str='ASC', before: str=None, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Optional page index for list pagination
        size: Optional page size for list pagination
        sort: Optional createdAt sort direction ASC or DESC
        before: Filter by created at before the given timestamp
        since: Filter by created at after the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/phone/numbers"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if sort:
        querystring['sort'] = sort
    if before:
        querystring['before'] = before
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallwebhookresults(size: int=20, sort: str='ASC', since: str=None, searchfilter: str=None, unseenonly: bool=None, page: int=0, before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    size: Optional page size in list pagination
        sort: Optional createdAt sort direction ASC or DESC
        since: Filter by created at after the given timestamp
        searchfilter: Optional search filter
        unseenonly: Filter for unseen exceptions only
        page: Optional page index in list pagination
        before: Filter by created at before the given timestamp
        
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/webhooks/results"
    querystring = {}
    if size:
        querystring['size'] = size
    if sort:
        querystring['sort'] = sort
    if since:
        querystring['since'] = since
    if searchfilter:
        querystring['searchFilter'] = searchfilter
    if unseenonly:
        querystring['unseenOnly'] = unseenonly
    if page:
        querystring['page'] = page
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getphoneplans(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/phone/plans"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportentities(outputformat: str, apikey: str, exporttype: str, listseparatortoken: str=None, excludepreviouslyexported: bool=None, createdoldesttime: str=None, createdearliesttime: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com/export"
    querystring = {'outputFormat': outputformat, 'apiKey': apikey, 'exportType': exporttype, }
    if listseparatortoken:
        querystring['listSeparatorToken'] = listseparatortoken
    if excludepreviouslyexported:
        querystring['excludePreviouslyExported'] = excludepreviouslyexported
    if createdoldesttime:
        querystring['createdOldestTime'] = createdoldesttime
    if createdearliesttime:
        querystring['createdEarliestTime'] = createdearliesttime
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jackmahoney-mailslurp-email-testing-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

