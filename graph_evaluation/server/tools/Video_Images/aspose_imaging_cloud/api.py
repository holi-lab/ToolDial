import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def modifywmf(pageheight: int, pagewidth: int, bordery: int, bkcolor: str, borderx: int, name: str, fromscratch: bool=None, storage: str=None, format: str='png', folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pageheight: Height of the page.
        pagewidth: Width of the page.
        bordery: Border height.
        bkcolor: Color of the background.
        borderx: Border width.
        name: Filename of image.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        storage: Your Aspose Cloud Storage name.
        format: Export format (PNG is the default one). Please, refer to the export table from https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        folder: Folder with image to process.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/wmf"
    querystring = {'pageHeight': pageheight, 'pageWidth': pagewidth, 'borderY': bordery, 'bkColor': bkcolor, 'borderX': borderx, }
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if storage:
        querystring['storage'] = storage
    if format:
        querystring['format'] = format
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convertimage(format: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Resulting image format. Please, refer to https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        name: Filename of image.
        folder: Folder with image to process.
        storage: Your Aspose Cloud Storage name.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/convert"
    querystring = {'format': format, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifyemf(name: str, bkcolor: str, borderx: int, bordery: int, pageheight: int, pagewidth: int, format: str='png', fromscratch: bool=None, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        bkcolor: Color of the background.
        borderx: Border width.
        bordery: Border height.
        pageheight: Height of the page.
        pagewidth: Width of the page.
        format: Export format (PNG is the default one). Please, refer to the export table from https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/emf"
    querystring = {'bkColor': bkcolor, 'borderX': borderx, 'borderY': bordery, 'pageHeight': pageheight, 'pageWidth': pagewidth, }
    if format:
        querystring['format'] = format
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifybmp(name: str, horizontalresolution: int, bitsperpixel: int, verticalresolution: int, storage: str=None, folder: str=None, fromscratch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        horizontalresolution: New horizontal resolution.
        bitsperpixel: Color depth.
        verticalresolution: New vertical resolution.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/bmp"
    querystring = {'horizontalResolution': horizontalresolution, 'bitsPerPixel': bitsperpixel, 'verticalResolution': verticalresolution, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifyjpeg(name: str, folder: str=None, compressiontype: str='baseline', quality: int=75, fromscratch: bool=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        folder: Folder with image to process.
        compressiontype: Compression type: baseline (default), progressive, lossless or jpegls.
        quality: Quality of an image from 0 to 100. Default is 75.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        storage: Your Aspose Cloud Storage name.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/jpg"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if compressiontype:
        querystring['compressionType'] = compressiontype
    if quality:
        querystring['quality'] = quality
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifypsd(name: str, folder: str=None, compressionmethod: str='rle', storage: str=None, channelscount: int=4, fromscratch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        folder: Folder with image to process.
        compressionmethod: Compression method (for now, raw and RLE are supported).
        storage: Your Aspose Cloud Storage name.
        channelscount: Count of color channels.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/psd"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if compressionmethod:
        querystring['compressionMethod'] = compressionmethod
    if storage:
        querystring['storage'] = storage
    if channelscount:
        querystring['channelsCount'] = channelscount
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cropimage(x: int, y: int, width: int, height: int, name: str, storage: str=None, folder: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    x: X position of start point for cropping rectangle.
        y: Y position of start point for cropping rectangle.
        width: Width of cropping rectangle
        height: Height of cropping rectangle.
        name: Filename of an image.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        format: Resulting image format. Please, refer to https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/crop"
    querystring = {'x': x, 'y': y, 'width': width, 'height': height, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageproperties(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of an image.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/properties"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageframerange(name: str, endframeid: int, startframeid: int, x: int=None, rectheight: int=None, newwidth: int=None, saveotherframes: bool=None, storage: str=None, y: int=None, rotateflipmethod: str=None, folder: str=None, newheight: int=None, rectwidth: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        endframeid: Index of the last frame in range.
        startframeid: Index of the first frame in range.
        x: X position of start point for cropping rectangle.
        rectheight: Height of cropping rectangle.
        newwidth: New width.
        saveotherframes: If result will include all other frames or just a specified frame.
        storage: Your Aspose Cloud Storage name.
        y: Y position of start point for cropping rectangle.
        rotateflipmethod: RotateFlip method (Rotate180FlipNone, Rotate180FlipX, Rotate180FlipXY, Rotate180FlipY, Rotate270FlipNone, Rotate270FlipX, Rotate270FlipXY, Rotate270FlipY, Rotate90FlipNone, Rotate90FlipX, Rotate90FlipXY, Rotate90FlipY, RotateNoneFlipNone, RotateNoneFlipX, RotateNoneFlipXY, RotateNoneFlipY). Default is RotateNoneFlipNone.
        folder: Folder with image to process.
        newheight: New height.
        rectwidth: Width of cropping rectangle.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/frames/range"
    querystring = {'endFrameId': endframeid, 'startFrameId': startframeid, }
    if x:
        querystring['x'] = x
    if rectheight:
        querystring['rectHeight'] = rectheight
    if newwidth:
        querystring['newWidth'] = newwidth
    if saveotherframes:
        querystring['saveOtherFrames'] = saveotherframes
    if storage:
        querystring['storage'] = storage
    if y:
        querystring['y'] = y
    if rotateflipmethod:
        querystring['rotateFlipMethod'] = rotateflipmethod
    if folder:
        querystring['folder'] = folder
    if newheight:
        querystring['newHeight'] = newheight
    if rectwidth:
        querystring['rectWidth'] = rectwidth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getavailablelabels(method: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    method: Object detection method
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/objectdetection/availablelabels/{method}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deskewimage(resizeproportionally: bool, name: str, bkcolor: str=None, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    resizeproportionally: Resize proportionally
        name: Image file name.
        bkcolor: Background color
        storage: Storage
        folder: Folder
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/deskew"
    querystring = {'resizeProportionally': resizeproportionally, }
    if bkcolor:
        querystring['bkColor'] = bkcolor
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifytiff(name: str, bitdepth: int, resolutionunit: str=None, folder: str=None, compression: str=None, storage: str=None, horizontalresolution: int=0, fromscratch: bool=None, verticalresolution: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        bitdepth: Bit depth.
        resolutionunit: New resolution unit (none - the default one, inch or centimeter).
        folder: Folder with image to process.
        compression: Compression (none is default). Please, refer to https://apireference.aspose.com/net/imaging/aspose.imaging.fileformats.tiff.enums/tiffcompressions for all possible values.
        storage: Your Aspose Cloud Storage name.
        horizontalresolution: New horizontal resolution.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        verticalresolution: New vertical resolution.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/tiff"
    querystring = {'bitDepth': bitdepth, }
    if resolutionunit:
        querystring['resolutionUnit'] = resolutionunit
    if folder:
        querystring['folder'] = folder
    if compression:
        querystring['compression'] = compression
    if storage:
        querystring['storage'] = storage
    if horizontalresolution:
        querystring['horizontalResolution'] = horizontalresolution
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if verticalresolution:
        querystring['verticalResolution'] = verticalresolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimagefeatures(searchcontextid: str, imageid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    searchcontextid: The search context identifier.
        imageid: The image identifier.
        storage: The storage.
        folder: The folder.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/features"
    querystring = {'imageId': imageid, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvisualobjectbounds(name: str, includescore: bool=None, method: str='ssd', color: str=None, threshold: int=50, storage: str=None, includelabel: bool=None, folder: str=None, blockedlabels: str=None, allowedlabels: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The image features detector.
        includescore: Draw detected objects scores
        method: Object detection method
        color: Bounds, labels, and scores text color
        threshold: Object detection probability threshold in percents
        storage: The storage.
        includelabel: Draw detected objects labels
        folder: The folder.
        blockedlabels: Comma-separated list of blocked labels
        allowedlabels: Comma-separated list of allowed labels
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/objectdetection/{name}/visualbounds"
    querystring = {}
    if includescore:
        querystring['includeScore'] = includescore
    if method:
        querystring['method'] = method
    if color:
        querystring['color'] = color
    if threshold:
        querystring['threshold'] = threshold
    if storage:
        querystring['storage'] = storage
    if includelabel:
        querystring['includeLabel'] = includelabel
    if folder:
        querystring['folder'] = folder
    if blockedlabels:
        querystring['blockedLabels'] = blockedlabels
    if allowedlabels:
        querystring['allowedLabels'] = allowedlabels
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findsimilarimages(similaritythreshold: int, maxcount: int, searchcontextid: str, imagedata: str=None, folder: str=None, imageid: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    similaritythreshold: The similarity threshold.
        maxcount: The maximum count.
        searchcontextid: The search context identifier.
        imagedata: Input image
        folder: The folder.
        imageid: The search image identifier.
        storage: The storage.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/findSimilar"
    querystring = {'similarityThreshold': similaritythreshold, 'maxCount': maxcount, }
    if imagedata:
        querystring['imageData'] = imagedata
    if folder:
        querystring['folder'] = folder
    if imageid:
        querystring['imageId'] = imageid
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifysvg(name: str, colortype: str='Rgb', folder: str=None, bordery: int=None, storage: str=None, bkcolor: str='white', format: str='svg', borderx: int=None, pagewidth: int=None, textasshapes: bool=None, scaley: int=0, pageheight: int=None, scalex: int=0, fromscratch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        colortype: Color type for SVG image. Only RGB is supported for now.
        folder: Folder with image to process.
        bordery: Border height. Only 0 is supported for now.
        storage: Your Aspose Cloud Storage name.
        bkcolor: Background color (Default is white).
        format: Export format (PNG is the default one). Please, refer to the export table from https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        borderx: Border width. Only 0 is supported for now.
        pagewidth: Width of the page.
        textasshapes: Whether text must be converted as shapes. true if all text is turned into SVG shapes in the convertion; otherwise, false
        scaley: Scale Y.
        pageheight: Height of the page.
        scalex: Scale X.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/svg"
    querystring = {}
    if colortype:
        querystring['colorType'] = colortype
    if folder:
        querystring['folder'] = folder
    if bordery:
        querystring['borderY'] = bordery
    if storage:
        querystring['storage'] = storage
    if bkcolor:
        querystring['bkColor'] = bkcolor
    if format:
        querystring['format'] = format
    if borderx:
        querystring['borderX'] = borderx
    if pagewidth:
        querystring['pageWidth'] = pagewidth
    if textasshapes:
        querystring['textAsShapes'] = textasshapes
    if scaley:
        querystring['scaleY'] = scaley
    if pageheight:
        querystring['pageHeight'] = pageheight
    if scalex:
        querystring['scaleX'] = scalex
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifygif(name: str, storage: str=None, backgroundcolorindex: int=32, hastrailer: bool=None, folder: str=None, fromscratch: bool=None, colorresolution: int=3, pixelaspectratio: int=3, ispalettesorted: bool=None, interlaced: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        storage: Your Aspose Cloud Storage name.
        backgroundcolorindex: Index of the background color.
        hastrailer: Specifies if image has trailer.
        folder: Folder with image to process.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        colorresolution: Color resolution.
        pixelaspectratio: Pixel aspect ratio.
        ispalettesorted: Specifies if palette is sorted.
        interlaced: Specifies if image is interlaced.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/gif"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if backgroundcolorindex:
        querystring['backgroundColorIndex'] = backgroundcolorindex
    if hastrailer:
        querystring['hasTrailer'] = hastrailer
    if folder:
        querystring['folder'] = folder
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if colorresolution:
        querystring['colorResolution'] = colorresolution
    if pixelaspectratio:
        querystring['pixelAspectRatio'] = pixelaspectratio
    if ispalettesorted:
        querystring['isPaletteSorted'] = ispalettesorted
    if interlaced:
        querystring['interlaced'] = interlaced
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def grayscaleimage(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Image file name.
        folder: Folder
        storage: Storage
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/grayscale"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getobjectbounds(name: str, storage: str=None, blockedlabels: str=None, method: str='ssd', includescore: bool=None, allowedlabels: str=None, threshold: int=50, includelabel: bool=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Image file name.
        storage: Storage
        blockedlabels: Comma-separated list of blocked labels
        method: Object detection method
        includescore: Return detected objects score
        allowedlabels: Comma-separated list of allowed labels
        threshold: Object detection probability threshold in percents
        includelabel: Return detected objects labels
        folder: Folder
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/objectdetection/{name}/bounds"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if blockedlabels:
        querystring['blockedLabels'] = blockedlabels
    if method:
        querystring['method'] = method
    if includescore:
        querystring['includeScore'] = includescore
    if allowedlabels:
        querystring['allowedLabels'] = allowedlabels
    if threshold:
        querystring['threshold'] = threshold
    if includelabel:
        querystring['includeLabel'] = includelabel
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resizeimage(newwidth: int, name: str, newheight: int, folder: str=None, storage: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    newwidth: New width.
        name: Filename of an image.
        newheight: New height.
        folder: Folder with image to process.
        storage: Your Aspose Cloud Storage name.
        format: Resulting image format. Please, refer to https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/resize"
    querystring = {'newWidth': newwidth, 'newHeight': newheight, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def updateimage(x: int, name: str, rectwidth: int, y: int, rectheight: int, newheight: int, newwidth: int, rotateflipmethod: str, storage: str=None, folder: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    x: X position of start point for cropping rectangle.
        name: Filename of an image.
        rectwidth: Width of cropping rectangle.
        y: Y position of start point for cropping rectangle.
        rectheight: Height of cropping rectangle.
        newheight: New height of the scaled image.
        newwidth: New width of the scaled image.
        rotateflipmethod: RotateFlip method (Rotate180FlipNone, Rotate180FlipX, Rotate180FlipXY, Rotate180FlipY, Rotate270FlipNone, Rotate270FlipX, Rotate270FlipXY, Rotate270FlipY, Rotate90FlipNone, Rotate90FlipX, Rotate90FlipXY, Rotate90FlipY, RotateNoneFlipNone, RotateNoneFlipX, RotateNoneFlipXY, RotateNoneFlipY). Default is RotateNoneFlipNone.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        format: Resulting image format. Please, refer to https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/updateImage"
    querystring = {'x': x, 'rectWidth': rectwidth, 'y': y, 'rectHeight': rectheight, 'newHeight': newheight, 'newWidth': newwidth, 'rotateFlipMethod': rotateflipmethod, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifywebp(name: str, animbackgroundcolor: str, quality: int, animloopcount: int, lossless: bool, storage: str=None, fromscratch: bool=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        animbackgroundcolor: Color of the animation background.
        quality: Quality (0-100).
        animloopcount: The animation loop count.
        lossless: If WEBP should be in lossless format.
        storage: Your Aspose Cloud Storage name.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        folder: Folder with image to process.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/webp"
    querystring = {'animBackgroundColor': animbackgroundcolor, 'quality': quality, 'animLoopCount': animloopcount, 'lossLess': lossless, }
    if storage:
        querystring['storage'] = storage
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearchimage(searchcontextid: str, imageid: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    searchcontextid: Search context identifier.
        imageid: Image identifier.
        folder: Folder.
        storage: Storage
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/image"
    querystring = {'imageId': imageid, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def converttifftofax(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        folder: Folder with image to process.
        storage: Your Aspose Cloud Storage name.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/tiff/{name}/toFax"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extractimagefeatures(imageid: str, searchcontextid: str, imagedata: str=None, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: The image identifier.
        searchcontextid: The search context identifier.
        imagedata: Input image
        folder: The folder.
        storage: The storage.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/image2features"
    querystring = {'imageId': imageid, }
    if imagedata:
        querystring['imageData'] = imagedata
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def modifyjpeg2000(name: str, comment: str, storage: str=None, fromscratch: bool=None, folder: str=None, codec: str='j2k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        comment: The comment (can be either single or comma-separated).
        storage: Your Aspose Cloud Storage name.
        fromscratch: Specifies where additional parameters we do not support should be taken from. If this is true – they will be taken from default values for standard image, if it is false – they will be saved from current image. Default is false.
        folder: Folder with image to process.
        codec: The codec (j2k or jp2).
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/jpg2000"
    querystring = {'comment': comment, }
    if storage:
        querystring['storage'] = storage
    if fromscratch:
        querystring['fromScratch'] = fromscratch
    if folder:
        querystring['folder'] = folder
    if codec:
        querystring['codec'] = codec
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findimageduplicates(searchcontextid: str, similaritythreshold: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    searchcontextid: The search context identifier.
        similaritythreshold: The similarity threshold.
        folder: The folder.
        storage: The storage.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/findDuplicates"
    querystring = {'similarityThreshold': similaritythreshold, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rotateflipimage(method: str, name: str, storage: str=None, folder: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    method: RotateFlip method (Rotate180FlipNone, Rotate180FlipX, Rotate180FlipXY, Rotate180FlipY, Rotate270FlipNone, Rotate270FlipX, Rotate270FlipXY, Rotate270FlipY, Rotate90FlipNone, Rotate90FlipX, Rotate90FlipXY, Rotate90FlipY, RotateNoneFlipNone, RotateNoneFlipX, RotateNoneFlipXY, RotateNoneFlipY).
        name: Filename of an image.
        storage: Your Aspose Cloud Storage name.
        folder: Folder with image to process.
        format: Resulting image format. Please, refer to https://docs.aspose.cloud/display/imagingcloud/Supported+File+Formats#SupportedFileFormats-CommonOperationsFormatSupportMap for possible use-cases.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/rotateflip"
    querystring = {'method': method, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        storagename: Storage name
        versionid: File version ID to download
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/file/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageframe(name: str, frameid: int, folder: str=None, x: int=None, rotateflipmethod: str=None, newheight: int=None, saveotherframes: bool=None, storage: str=None, newwidth: int=None, rectwidth: int=None, rectheight: int=None, y: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename of image.
        frameid: Number of a frame.
        folder: Folder with image to process.
        x: X position of start point for cropping rectangle.
        rotateflipmethod: RotateFlip method (Rotate180FlipNone, Rotate180FlipX, Rotate180FlipXY, Rotate180FlipY, Rotate270FlipNone, Rotate270FlipX, Rotate270FlipXY, Rotate270FlipY, Rotate90FlipNone, Rotate90FlipX, Rotate90FlipXY, Rotate90FlipY, RotateNoneFlipNone, RotateNoneFlipX, RotateNoneFlipXY, RotateNoneFlipY). Default is RotateNoneFlipNone.
        newheight: New height.
        saveotherframes: If result will include all other frames or just a specified frame.
        storage: Your Aspose Cloud Storage name.
        newwidth: New width.
        rectwidth: Width of cropping rectangle.
        rectheight: Height of cropping rectangle.
        y: Y position of start point for cropping rectangle.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/frames/{frameid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if x:
        querystring['x'] = x
    if rotateflipmethod:
        querystring['rotateFlipMethod'] = rotateflipmethod
    if newheight:
        querystring['newHeight'] = newheight
    if saveotherframes:
        querystring['saveOtherFrames'] = saveotherframes
    if storage:
        querystring['storage'] = storage
    if newwidth:
        querystring['newWidth'] = newwidth
    if rectwidth:
        querystring['rectWidth'] = rectwidth
    if rectheight:
        querystring['rectHeight'] = rectheight
    if y:
        querystring['y'] = y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectexists(path: str, versionid: str=None, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        versionid: File version ID
        storagename: Storage name
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/storage/exist/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimagesearchstatus(searchcontextid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    searchcontextid: The search context identifier.
        storage: The storage.
        folder: The folder.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/ai/imageSearch/{searchcontextid}/status"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageframeproperties(name: str, frameid: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Filename with image.
        frameid: Number of a frame.
        folder: Folder with image to process.
        storage: Your Aspose Cloud Storage name.
        
    """
    url = f"https://aspose-imaging-cloud1.p.rapidapi.com/imaging/{name}/frames/{frameid}/properties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-imaging-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

