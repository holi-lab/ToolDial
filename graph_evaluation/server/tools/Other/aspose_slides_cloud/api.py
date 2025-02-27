import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getslidesdocument(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        storage: Documentstorage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesapiinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideanimation(slideindex: int, name: str, shapeindex: int=None, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        shapeindex: Shape index. If specified, only effects related to that shape are returned.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/animation"
    querystring = {}
    if shapeindex:
        querystring['shapeIndex'] = shapeindex
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphportion(portionindex: int, paragraphindex: int, slideindex: int, shapeindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    portionindex: Portion index.
        paragraphindex: Paragraph index.
        slideindex: Slide index.
        shapeindex: Shape index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{shapeindex}/paragraphs/{paragraphindex}/portions/{portionindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideheaderfooter(slideindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: The position of the slide to be reordered.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/headerFooter"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesthemefontscheme(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/theme/fontScheme"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectexists(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        storagename: Storage name
        versionid: File version ID
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/exist/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlayoutslide(name: str, slideindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/layoutSlides/{slideindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesimages(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/images"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidestheme(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/theme"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshapes(slideindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshapeportion(shapeindex: int, portionindex: int, paragraphindex: int, name: str, slideindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    shapeindex: Shape index.
        portionindex: Portion index.
        paragraphindex: Paragraph index.
        name: Document name.
        slideindex: Slide index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes/{shapeindex}/paragraphs/{paragraphindex}/portions/{portionindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideexists(slideindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/exist"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesdocumentproperties(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/documentproperties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslide(slideindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesplaceholder(name: str, slideindex: int, placeholderindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        placeholderindex: Placeholder index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/placeholders/{placeholderindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmasterslideslist(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/masterSlides"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesplaceholders(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/placeholders"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesprotectionproperties(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/protectionProperties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshapeparagraphs(slideindex: int, shapeindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        shapeindex: Shape index.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes/{shapeindex}/paragraphs"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlayoutslideslist(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/layoutSlides"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshapeparagraph(shapeindex: int, slideindex: int, name: str, paragraphindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    shapeindex: Shape index.
        slideindex: Slide index.
        name: Document name.
        paragraphindex: Paragraph index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes/{shapeindex}/paragraphs/{paragraphindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmasterslide(name: str, slideindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/masterSlides/{slideindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslideimages(name: str, slideindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/images"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesimagewithformat(format: str, name: str, index: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Export format (png, jpg, gif).
        name: Document name.
        index: Image index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/images/{index}/{format}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshape(shapeindex: int, slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    shapeindex: Shape index.
        slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes/{shapeindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslidewithformat(slideindex: int, name: str, format: str, height: int=None, width: int=None, fontsfolder: str=None, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        format: Output file format.
        height: Output file height.
        width: Output file width.
        fontsfolder: Storage folder containing custom fonts to be used with the document.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/{format}"
    querystring = {}
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if fontsfolder:
        querystring['fontsFolder'] = fontsfolder
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideheaderfooter(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/headerFooter"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesviewproperties(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/viewProperties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotesslideshapeportions(slideindex: int, name: str, paragraphindex: int, shapeindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        paragraphindex: Paragraph index.
        shapeindex: Shape index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/notesSlide/shapes/{shapeindex}/paragraphs/{paragraphindex}/portions"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesdocumentproperty(propertyname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    propertyname: The property name.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/documentproperties/{propertyname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideshapeparagraph(slideindex: int, name: str, paragraphindex: int, shapeindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        paragraphindex: Paragraph index.
        shapeindex: Shape index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{shapeindex}/paragraphs/{paragraphindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideshape(slideindex: int, shapeindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        shapeindex: Shape index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{shapeindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideshapes(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesubshape(slideindex: int, shapeindex: int, path: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        shapeindex: Shape index.
        path: Shape path.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}/{shapeindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslideproperties(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slideProperties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesimagewithdefaultformat(name: str, index: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        index: Image index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/images/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparagraphportions(paragraphindex: int, slideindex: int, name: str, shapeindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    paragraphindex: Paragraph index.
        slideindex: Slide index.
        name: Document name.
        shapeindex: Shape index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{shapeindex}/paragraphs/{paragraphindex}/portions"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsections(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/sections"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslidebackground(name: str, slideindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/background"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdiscusage(storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslideshapeparagraphs(slideindex: int, name: str, shapeindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        shapeindex: Shape index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{shapeindex}/paragraphs"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesubshapeparagraph(name: str, paragraphindex: int, slideindex: int, path: str, shapeindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        paragraphindex: Paragraph index.
        slideindex: Slide index.
        path: Shape path.
        shapeindex: Shape index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}/{shapeindex}/paragraphs/{paragraphindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesthemecolorscheme(name: str, slideindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/theme/colorScheme"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubshapeparagraphportions(paragraphindex: int, path: str, name: str, slideindex: int, shapeindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    paragraphindex: Paragraph index.
        path: Shape path.
        name: Document name.
        slideindex: Slide index.
        shapeindex: Shape index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}/{shapeindex}/paragraphs/{paragraphindex}/portions"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslidecomments(slideindex: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: The position of the slide to be reordered.
        name: Document name.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/comments"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storageexists(storagename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(path: str, versionid: str=None, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        versionid: File version ID to download
        storagename: Storage name
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/file/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslidetextitems(slideindex: int, name: str, storage: str=None, withempty: bool=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        storage: Document storage.
        withempty: True to include empty items.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/textItems"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if withempty:
        querystring['withEmpty'] = withempty
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidespresentationtextitems(name: str, withempty: bool=None, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        withempty: True to incude empty items.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/textItems"
    querystring = {}
    if withempty:
        querystring['withEmpty'] = withempty
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesthemeformatscheme(name: str, slideindex: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/theme/formatScheme"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubshapeparagraphportion(path: str, name: str, slideindex: int, paragraphindex: int, portionindex: int, shapeindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: Shape path.
        name: Document name.
        slideindex: Slide index.
        paragraphindex: Paragraph index.
        portionindex: Portion index.
        shapeindex: Shape index.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}/{shapeindex}/paragraphs/{paragraphindex}/portions/{portionindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesubshapeparagraphs(slideindex: int, name: str, shapeindex: int, path: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        shapeindex: Shape index.
        path: Shape path.
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}/{shapeindex}/paragraphs"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileslist(path: str, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: Folder path e.g. '/folder'
        storagename: Storage name
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslideslist(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesslide(slideindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    slideindex: Slide index.
        name: Document name.
        folder: Document folder.
        storage: Document storage.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getslidesubshapes(name: str, slideindex: int, path: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Document name.
        slideindex: Slide index.
        path: Shape path (for smart art and group shapes).
        storage: Document storage.
        folder: Document folder.
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/{name}/slides/{slideindex}/shapes/{path}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileversions(path: str, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/file.ext'
        storagename: Storage name
        
    """
    url = f"https://aspose-slides-cloud.p.rapidapi.com/slides/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-slides-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

