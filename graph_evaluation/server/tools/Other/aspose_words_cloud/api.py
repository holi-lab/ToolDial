import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def downloadfile(path: str, storagename: str='string', versionid: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download file"
    path: Path of the file including the file name and extension e.g. /folder1/file.ext
        storagename: Storage name
        versionid: File version ID to download
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/storage/file/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getformfields(nodepath: str, name: str, password: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets form fields from document."
    nodepath: Path to the node containing collection of form fields.
        name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/formfields"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getheaderfooters(sectionpath: str, name: str, folder: str='string', filterbytype: str='string', loadencoding: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of header/footers from the document."
    sectionpath: Path to parent section.
        name: The document name.
        folder: Original document folder.
        filterbytype: List of types of headers and footers.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{sectionpath}/headersfooters"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if filterbytype:
        querystring['filterByType'] = filterbytype
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfootnote(index: int, nodepath: str, name: str, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads footnote by index."
    index: Object index.
        nodepath: Path to the node, which contains collection of footnotes.
        name: The document name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/footnotes/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getofficemathobject(index: int, name: str, nodepath: str, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads OfficeMath object by index."
    index: Object index.
        name: The document name.
        nodepath: Path to the node, which contains collection of OfficeMath objects.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/OfficeMathObjects/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renderpage(pageindex: int, format: str, name: str, storage: str='string', folder: str='string', fontslocation: str='string', loadencoding: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders page to specified format."
    pageindex: Comment index.
        format: The destination format.
        name: The document name.
        storage: Original document storage.
        folder: Original document folder.
        fontslocation: Folder in filestorage with custom fonts.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/pages/{pageindex}/render"
    querystring = {'format': format, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrangetext(rangeendidentifier: str, name: str, rangestartidentifier: str, folder: str='string', loadencoding: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the text from the range."
    rangeendidentifier: The range end identifier.
        name: The document.
        rangestartidentifier: The range start identifier.
Identifier is the value of the "nodeId" field, which every document node has, extended with the prefix "id".
It looks like "id0.7". Also values like "image5" and "table3" can be used as an identifier for images and tables, where the number is an index of the image/table.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/range/{rangestartidentifier}/{rangeendidentifier}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfield(nodepath: str, name: str, index: int, loadencoding: str='string', folder: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets field from document."
    nodepath: Path to the node, which contains collection of fields.
        name: The document name.
        index: Object index.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/fields/{index}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfieldnames(name: str, usenonmergefields: bool=None, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document field names."
    name: The document name.
        usenonmergefields: If true, result includes "mustache" field names.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/mailMerge/FieldNames"
    querystring = {}
    if usenonmergefields:
        querystring['useNonMergeFields'] = usenonmergefields
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getformfield(name: str, index: int, nodepath: str, password: str='string', loadencoding: str='string', folder: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns representation of an one of the form field."
    name: The document name.
        index: Object index.
        nodepath: Path to the node that contains collection of formfields.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/formfields/{index}"
    querystring = {}
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getavailablefonts(fontslocation: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of fonts, available for document processing."
    fontslocation: Folder in filestorage with custom fonts.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/fonts/available"
    querystring = {}
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfootnotes(name: str, nodepath: str, folder: str='string', loadencoding: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets footnotes from document."
    name: The document name.
        nodepath: Path to the node, which contains collection of footnotes.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/footnotes"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getheaderfooterofsection(sectionindex: int, headerfooterindex: int, name: str, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', filterbytype: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a header/footer from the document section."
    sectionindex: Section index.
        headerfooterindex: Header/footer index.
        name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        filterbytype: List of types of headers and footers.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/sections/{sectionindex}/headersfooters/{headerfooterindex}"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if filterbytype:
        querystring['filterByType'] = filterbytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rendermathobject(format: str, nodepath: str, name: str, index: int, password: str='string', loadencoding: str='string', fontslocation: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders math object to specified format."
    format: The destination format.
        nodepath: Path to the node, which contains office math objects.
        name: The document name.
        index: Object index.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        fontslocation: Folder in filestorage with custom fonts.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/OfficeMathObjects/{index}/render"
    querystring = {'format': format, }
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablerowformat(tablepath: str, index: int, name: str, password: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table row format."
    tablepath: Path to table.
        index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{tablepath}/rows/{index}/rowformat"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenthyperlinks(name: str, loadencoding: str='string', password: str='string', folder: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document hyperlinks common info."
    name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/hyperlinks"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenthyperlinkbyindex(name: str, hyperlinkindex: int, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document hyperlink by its index."
    name: The document name.
        hyperlinkindex: The hyperlink index.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/hyperlinks/{hyperlinkindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablecellformat(tablerowpath: str, index: int, name: str, password: str='string', storage: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table cell format."
    tablerowpath: Path to table row.
        index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{tablerowpath}/cells/{index}/cellformat"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getofficemathobjects(name: str, nodepath: str, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets OfficeMath objects from document."
    name: The document name.
        nodepath: Path to the node, which contains collection of OfficeMath objects.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/OfficeMathObjects"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrun(index: int, paragraphpath: str, name: str, folder: str='string', loadencoding: str='string', password: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents run of text contained in the document."
    index: Object index.
        paragraphpath: Path to parent paragraph.
        name: The document name.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{paragraphpath}/runs/{index}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getruns(paragraphpath: str, name: str, password: str='string', folder: str='string', storage: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents collection of runs in the paragraph."
    paragraphpath: Path to parent paragraph.
        name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{paragraphpath}/runs"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettables(nodepath: str, name: str, folder: str='string', password: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of tables that are contained in the document."
    nodepath: Path to the node, which contains tables.
        name: The document name.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/tables"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraph(index: int, name: str, nodepath: str, storage: str='string', folder: str='string', loadencoding: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents one of the paragraphs contained in the document."
    index: Object index.
        name: The document name.
        nodepath: Path to the node which contains paragraphs.
        storage: Original document storage.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/paragraphs/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsections(name: str, password: str='string', storage: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of sections that are contained in the document."
    name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/sections"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphformat(nodepath: str, index: int, name: str, password: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Represents all the formatting for a paragraph."
    nodepath: Path to the node which contains paragraphs.
        index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/paragraphs/{index}/format"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentstatistics(name: str, includetextinshapes: bool=None, includecomments: bool=None, folder: str='string', password: str='string', loadencoding: str='string', includefootnotes: bool=None, storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document statistics."
    name: The document name.
        includetextinshapes: Support including/excluding shape's text from the WordCount. Default value is "false".
        includecomments: Support including/excluding comments from the WordCount. Default value is "false".
        folder: Original document folder.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        includefootnotes: Support including/excluding footnotes from the WordCount. Default value is "false".
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/statistics"
    querystring = {}
    if includetextinshapes:
        querystring['includeTextInShapes'] = includetextinshapes
    if includecomments:
        querystring['includeComments'] = includecomments
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if includefootnotes:
        querystring['includeFootnotes'] = includefootnotes
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renderparagraph(name: str, index: int, format: str, nodepath: str, password: str='string', loadencoding: str='string', fontslocation: str='string', folder: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders paragraph to specified format."
    name: The document name.
        index: Object index.
        format: The destination format.
        nodepath: Path to the node, which contains paragraphs.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        fontslocation: Folder in filestorage with custom fonts.
        folder: Original document folder.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/paragraphs/{index}/render"
    querystring = {'format': format, }
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphs(name: str, nodepath: str, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of paragraphs that are contained in the document."
    name: The document name.
        nodepath: Path to the node which contains paragraphs.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/paragraphs"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrunfont(index: int, paragraphpath: str, name: str, folder: str='string', loadencoding: str='string', password: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents font of run."
    index: Object index.
        paragraphpath: Path to parent paragraph.
        name: The document name.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{paragraphpath}/runs/{index}/font"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileslist(path: str, storagename: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all files and folders within a folder"
    path: Folder path e.g. /Folder1
        storagename: Storage name
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getformfieldswithoutnodepath(name: str, loadencoding: str='string', folder: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets form fields from document."
    name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/formfields"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablecell(name: str, tablerowpath: str, index: int, password: str='string', loadencoding: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table cell."
    name: The document name.
        tablerowpath: Path to table row.
        index: Object index.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{tablerowpath}/cells/{index}"
    querystring = {}
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(pattern: str, name: str, loadencoding: str='string', storage: str='string', folder: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches text in document."
    pattern: The regular expression used to find matches.
        name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/search"
    querystring = {'pattern': pattern, }
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfootnotewithoutnodepath(name: str, index: int, loadencoding: str='string', folder: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads footnote by index."
    name: The document name.
        index: Object index.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/footnotes/{index}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsection(name: str, sectionindex: int, folder: str='string', storage: str='string', loadencoding: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets document section by index."
    name: The document name.
        sectionindex: Section index.
        folder: Original document folder.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/sections/{sectionindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getofficemathobjectswithoutnodepath(name: str, folder: str='string', loadencoding: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets OfficeMath objects from document."
    name: The document name.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/OfficeMathObjects"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getformfieldwithoutnodepath(name: str, index: int, password: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns representation of an one of the form field."
    name: The document name.
        index: Object index.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/formfields/{index}"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rendertablewithoutnodepath(name: str, index: int, format: str, fontslocation: str='string', loadencoding: str='string', storage: str='string', password: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders table to specified format."
    name: The document name.
        index: Object index.
        format: The destination format.
        fontslocation: Folder in filestorage with custom fonts.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/tables/{index}/render"
    querystring = {'format': format, }
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rendermathobjectwithoutnodepath(name: str, format: str, index: int, folder: str='string', fontslocation: str='string', storage: str='string', password: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders math object to specified format."
    name: The document name.
        format: The destination format.
        index: Object index.
        folder: Original document folder.
        fontslocation: Folder in filestorage with custom fonts.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/OfficeMathObjects/{index}/render"
    querystring = {'format': format, }
    if folder:
        querystring['folder'] = folder
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablepropertieswithoutnodepath(name: str, index: int, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table properties."
    name: The document name.
        index: Object index.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/tables/{index}/properties"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfieldwithoutnodepath(name: str, index: int, loadencoding: str='string', password: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets field from document."
    name: The document name.
        index: Object index.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/fields/{index}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsectionpagesetup(name: str, sectionindex: int, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets page setup of section."
    name: The document name.
        sectionindex: Section index.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/sections/{sectionindex}/pageSetup"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablerow(tablepath: str, name: str, index: int, storage: str='string', loadencoding: str='string', password: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table row."
    tablepath: Path to table.
        name: The document name.
        index: Object index.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{tablepath}/rows/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphformatwithoutnodepath(name: str, index: int, loadencoding: str='string', storage: str='string', folder: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Represents all the formatting for a paragraph."
    name: The document name.
        index: Object index.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/paragraphs/{index}/format"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rendertable(nodepath: str, format: str, index: int, name: str, password: str='string', fontslocation: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders table to specified format."
    nodepath: Path to the node, which contains tables.
        format: The destination format.
        index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        fontslocation: Folder in filestorage with custom fonts.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/tables/{index}/render"
    querystring = {'format': format, }
    if password:
        querystring['password'] = password
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettableproperties(index: int, nodepath: str, name: str, password: str='string', folder: str='string', storage: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table properties."
    index: Object index.
        nodepath: Path to the node, which contains tables.
        name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/tables/{index}/properties"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getofficemathobjectwithoutnodepath(index: int, name: str, folder: str='string', password: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads OfficeMath object by index."
    index: Object index.
        name: The document name.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/OfficeMathObjects/{index}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphswithoutnodepath(name: str, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of paragraphs that are contained in the document."
    name: The document name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/paragraphs"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettableswithoutnodepath(name: str, password: str='string', folder: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of tables that are contained in the document."
    name: The document name.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/tables"
    querystring = {}
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettable(nodepath: str, index: int, name: str, loadencoding: str='string', password: str='string', folder: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table."
    nodepath: Path to the node, which contains tables.
        index: Object index.
        name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/tables/{index}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettablewithoutnodepath(name: str, index: int, folder: str='string', storage: str='string', password: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a table."
    name: The document name.
        index: Object index.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/tables/{index}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfootnoteswithoutnodepath(name: str, storage: str='string', folder: str='string', loadencoding: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets footnotes from document."
    name: The document name.
        storage: Original document storage.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/footnotes"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocument(documentname: str, folder: str='string', password: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document common info."
    documentname: The document name.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{documentname}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getheaderfooter(name: str, headerfooterindex: int, loadencoding: str='string', password: str='string', storage: str='string', folder: str='string', filterbytype: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a header/footer from the document by index."
    name: The document name.
        headerfooterindex: Header/footer index.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        folder: Original document folder.
        filterbytype: List of types of headers and footers.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/headersfooters/{headerfooterindex}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if filterbytype:
        querystring['filterByType'] = filterbytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renderparagraphwithoutnodepath(index: int, format: str, name: str, folder: str='string', password: str='string', fontslocation: str='string', loadencoding: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders paragraph to specified format."
    index: Object index.
        format: The destination format.
        name: The document name.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        fontslocation: Folder in filestorage with custom fonts.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/paragraphs/{index}/render"
    querystring = {'format': format, }
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renderdrawingobjectwithoutnodepath(index: int, format: str, name: str, storage: str='string', fontslocation: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders drawing object to specified format."
    index: Object index.
        format: The destination format.
        name: The document name.
        storage: Original document storage.
        fontslocation: Folder in filestorage with custom fonts.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/drawingObjects/{index}/render"
    querystring = {'format': format, }
    if storage:
        querystring['storage'] = storage
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphwithoutnodepath(index: int, name: str, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents one of the paragraphs contained in the document."
    index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/paragraphs/{index}"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentprotection(name: str, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document protection common info."
    name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/protection"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookmarkbyname(bookmarkname: str, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document bookmark data by its name."
    bookmarkname: The bookmark name.
        name: The document name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/bookmarks/{bookmarkname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookmarks(name: str, folder: str='string', storage: str='string', password: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document bookmarks common info."
    name: The document name.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/bookmarks"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getborder(nodepath: str, index: int, name: str, loadencoding: str='string', storage: str='string', password: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "'nodePath' should refer to node with cell or row."
    nodepath: Path to the node with border(node should be cell or row).
        index: Object index.
        name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/borders/{index}"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getborders(nodepath: str, name: str, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "'nodePath' should refer to node with cell or row."
    nodepath: Path to the node with borders (node should be cell or row).
        name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/borders"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classifydocument(documentname: str, bestclassescount: str='string', taxonomy: str='string', loadencoding: str='string', folder: str='string', password: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Classifies document."
    documentname: The document name.
        bestclassescount: Count of the best classes to return.
        taxonomy: Taxonomy to use for classification return.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{documentname}/classify"
    querystring = {}
    if bestclassescount:
        querystring['bestClassesCount'] = bestclassescount
    if taxonomy:
        querystring['taxonomy'] = taxonomy
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcomment(name: str, commentindex: int, password: str='string', loadencoding: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets comment from document."
    name: The document name.
        commentindex: The comment index.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/comments/{commentindex}"
    querystring = {}
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcomments(name: str, loadencoding: str='string', storage: str='string', folder: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets comments from document."
    name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/comments"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentwithformat(name: str, format: str, loadencoding: str='string', password: str='string', fontslocation: str='string', storage: str='string', folder: str='string', outpath: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exports the document into the specified format."
    name: The document name.
        format: The destination format.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        fontslocation: Folder in filestorage with custom fonts.
        storage: Original document storage.
        folder: Original document folder.
        outpath: Path to save the result.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}"
    querystring = {'format': format, }
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if fontslocation:
        querystring['fontsLocation'] = fontslocation
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if outpath:
        querystring['outPath'] = outpath
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperties(name: str, storage: str='string', password: str='string', folder: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document properties info."
    name: The document's name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/documentProperties"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperty(name: str, propertyname: str, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document property info by the property name."
    name: The document name.
        propertyname: The property name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/documentProperties/{propertyname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectbyindex(nodepath: str, name: str, index: int, folder: str='string', storage: str='string', password: str='string', loadencoding: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document drawing object common info by its index or convert to format specified."
    nodepath: Path to the node, which contains collection of drawing objects.
        name: The document name.
        index: Object index.
        folder: Original document folder.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/drawingObjects/{index}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjects(nodepath: str, name: str, storage: str='string', folder: str='string', loadencoding: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document drawing objects common info."
    nodepath: Path to the node, which contains collection of drawing objects.
        name: The document name.
        storage: Original document storage.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/drawingObjects"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectoledata(index: int, name: str, nodepath: str, folder: str='string', loadencoding: str='string', storage: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets drawing object OLE data."
    index: Object index.
        name: The document name.
        nodepath: Path to the node, which contains collection of drawing objects.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/drawingObjects/{index}/oleData"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectbyindexwithoutnodepath(name: str, index: int, storage: str='string', loadencoding: str='string', folder: str='string', password: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document drawing object common info by its index or convert to format specified."
    name: The document name.
        index: Object index.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        password: Password for opening an encrypted document.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/drawingObjects/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectoledatawithoutnodepath(index: int, name: str, password: str='string', storage: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets drawing object OLE data."
    index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/drawingObjects/{index}/oleData"
    querystring = {}
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectimagedatawithoutnodepath(name: str, index: int, folder: str='string', loadencoding: str='string', password: str='string', storage: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads drawing object image data."
    name: The document name.
        index: Object index.
        folder: Original document folder.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/drawingObjects/{index}/imageData"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectswithoutnodepath(name: str, password: str='string', loadencoding: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads document drawing objects common info."
    name: The document name.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/drawingObjects"
    querystring = {}
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdrawingobjectimagedata(nodepath: str, index: int, name: str, password: str='string', loadencoding: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reads drawing object image data."
    nodepath: Path to the node, which contains collection of drawing objects.
        index: Object index.
        name: The document name.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/drawingObjects/{index}/imageData"
    querystring = {}
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renderdrawingobject(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renders drawing object to specified format."
    
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodePath}/drawingObjects/{index}/render"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfieldswithoutnodepath(name: str, storage: str='string', password: str='string', loadencoding: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fields from document."
    name: The document name.
        storage: Original document storage.
        password: Password for opening an encrypted document.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/fields"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if password:
        querystring['password'] = password
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfields(nodepath: str, name: str, loadencoding: str='string', password: str='string', storage: str='string', folder: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fields from document."
    nodepath: Path to the node, which contains collection of fields.
        name: The document name.
        loadencoding: Encoding that will be used to load an HTML (or TXT) document if the encoding is not specified in HTML.
        password: Password for opening an encrypted document.
        storage: Original document storage.
        folder: Original document folder.
        
    """
    url = f"https://aspose-words-cloud1.p.rapidapi.com/words/{name}/{nodepath}/fields"
    querystring = {}
    if loadencoding:
        querystring['loadEncoding'] = loadencoding
    if password:
        querystring['password'] = password
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-words-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

