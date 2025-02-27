import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def downloadattachment(documentid: str, attachmentid: str, onbehalfof: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    documentid: Document Id.
        attachmentid: Attachment Id(Get attachment ID from Properties API).
        onbehalfof: The on behalfof email address.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/downloadAttachment"
    querystring = {'documentId': documentid, 'attachmentId': attachmentid, }
    if onbehalfof:
        querystring['onBehalfOf'] = onbehalfof
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listdocuments(page: int, labels: str='[]', transmittype: str='Sent', recipients: str='[]', startdate: str=None, enddate: str=None, status: str='[
  "None"
]', pagesize: int=10, searchkey: str=None, sentby: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/list"
    querystring = {'Page': page, }
    if labels:
        querystring['Labels'] = labels
    if transmittype:
        querystring['TransmitType'] = transmittype
    if recipients:
        querystring['Recipients'] = recipients
    if startdate:
        querystring['StartDate'] = startdate
    if enddate:
        querystring['EndDate'] = enddate
    if status:
        querystring['Status'] = status
    if pagesize:
        querystring['PageSize'] = pagesize
    if searchkey:
        querystring['SearchKey'] = searchkey
    if sentby:
        querystring['SentBy'] = sentby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def behalfdocuments(page: int, searchkey: str=None, pagesize: int=10, startdate: str=None, labels: str='[]', status: str='[
  "None"
]', enddate: str=None, pagetype: str='BehalfOfOthers', signers: str='[]', emailaddress: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page index specified in get document list request (cursor).
        searchkey: The search key used to filter the documents returned in the API. The API will return documents that contain the search key in the document title, document ID, sender or signer(s) name, etc.
        pagesize: Page size specified in get document list request.
        startdate: The start date used to filter the documents returned in the API. The API will return documents that were created on or after this date
        labels: A list of labels or tags used to filter the documents returned in the API. The API will return documents that have been tagged with one or more of the labels provided in this list
        status: Filter documents based on their current status, including In-progress, Completed, Declined, Expired, and Revoked.
        enddate: The end date used to filter the documents returned in the API. The API will return documents that were created on or before this date.
        pagetype: The filter used to differentiate between documents sent on the user's behalf and documents sent by the user on behalf of others. The API will return documents based on the specified value.
        signers: A list of signer email addresses used to filter the documents returned in the API. The API will return documents where the signer's email address matches one of the email addresses provided in this list
        emailaddress: The sender identity's email used to filter the documents returned in the API. The API will return documents that were sent on behalf of the specified email address.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/behalfList"
    querystring = {'Page': page, }
    if searchkey:
        querystring['SearchKey'] = searchkey
    if pagesize:
        querystring['PageSize'] = pagesize
    if startdate:
        querystring['StartDate'] = startdate
    if labels:
        querystring['Labels'] = labels
    if status:
        querystring['Status'] = status
    if enddate:
        querystring['EndDate'] = enddate
    if pagetype:
        querystring['PageType'] = pagetype
    if signers:
        querystring['Signers'] = signers
    if emailaddress:
        querystring['EmailAddress'] = emailaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloaddocument(documentid: str, onbehalfof: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    documentid: Document Id.
        onbehalfof: The on behalfof email address.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/download"
    querystring = {'documentId': documentid, }
    if onbehalfof:
        querystring['onBehalfOf'] = onbehalfof
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getembeddedsignlink(signeremail: str, documentid: str, signlinkvalidtill: str=None, redirecturl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/getEmbeddedSignLink"
    querystring = {'SignerEmail': signeremail, 'DocumentId': documentid, }
    if signlinkvalidtill:
        querystring['SignLinkValidTill'] = signlinkvalidtill
    if redirecturl:
        querystring['RedirectUrl'] = redirecturl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproperties(documentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    documentid: Document Id.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/properties"
    querystring = {'documentId': documentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teamdocuments(page: int, enddate: str=None, labels: str='[]', teamid: str='[]', transmittype: str='Sent', searchkey: str=None, pagesize: int=10, startdate: str=None, userid: str='[]', status: str='[
  "None"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page index specified in get document list request.
        enddate: End date of the document
        labels: Labels of the document.
        teamid: TeamId  of the  Team document.
        transmittype: Transmit type as Sent, Received and Both.
        searchkey: Documents can be listed by the search key present in the document like document title, document ID, sender or recipient(s) name, etc.,
        pagesize: Page size specified in get document list request.
        startdate: Start date of the document
        userid: UserId of the  Team document.
        status: Status of the document such as In-progress, Completed, Decline, Expired, Revoked, Draft.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/teamlist"
    querystring = {'Page': page, }
    if enddate:
        querystring['EndDate'] = enddate
    if labels:
        querystring['Labels'] = labels
    if teamid:
        querystring['TeamId'] = teamid
    if transmittype:
        querystring['TransmitType'] = transmittype
    if searchkey:
        querystring['SearchKey'] = searchkey
    if pagesize:
        querystring['PageSize'] = pagesize
    if startdate:
        querystring['StartDate'] = startdate
    if userid:
        querystring['UserId'] = userid
    if status:
        querystring['Status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_plan_apicreditscount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://boldsign.p.rapidapi.com/v1/plan/apiCreditsCount"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listsenderidentities(page: int, search: str=None, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page index specified in get sender identity request.
        search: Users can be listed by the search key present in the sender identity like sender identity name and email address
        pagesize: Page size specified in get sender identity list request.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/senderIdentities/list"
    querystring = {'Page': page, }
    if search:
        querystring['Search'] = search
    if pagesize:
        querystring['PageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listusers(page: int, pagesize: int=10, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page index specified in get user list request.
        pagesize: Page size specified in get user list request.
        search: Users can be listed by the search  based on the user ID
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/users/list"
    querystring = {'Page': page, }
    if pagesize:
        querystring['PageSize'] = pagesize
    if search:
        querystring['Search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuser(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    userid: User Id.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/users/get"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brandlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://boldsign.p.rapidapi.com/v1/brand/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteam(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: Team Id.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/teams/get"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listteams(page: int, pagesize: int=10, searchkey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page index specified in get team list request.
        pagesize: Page size specified in get team list request.
        searchkey: Teams can be listed by the search key
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/teams/list"
    querystring = {'Page': page, }
    if pagesize:
        querystring['PageSize'] = pagesize
    if searchkey:
        querystring['SearchKey'] = searchkey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtemplates(page: int, searchkey: str=None, templatetype: str='mytemplates', pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://boldsign.p.rapidapi.com/v1/template/list"
    querystring = {'Page': page, }
    if searchkey:
        querystring['SearchKey'] = searchkey
    if templatetype:
        querystring['TemplateType'] = templatetype
    if pagesize:
        querystring['PageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download(templateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    templateid: Template Id.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/template/download"
    querystring = {'templateId': templateid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadauditlog(documentid: str, onbehalfof: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    documentid: Document Id.
        onbehalfof: The on behalfof email address.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/document/downloadAuditLog"
    querystring = {'documentId': documentid, }
    if onbehalfof:
        querystring['onBehalfOf'] = onbehalfof
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproperties(templateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    templateid: Template Id.
        
    """
    url = f"https://boldsign.p.rapidapi.com/v1/template/properties"
    querystring = {'templateId': templateid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boldsign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

