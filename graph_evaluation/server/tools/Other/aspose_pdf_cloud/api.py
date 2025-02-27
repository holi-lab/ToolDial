import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdocumentpopupannotationsbyparent(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The parent annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/{annotationid}/popup"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfreetextannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/freetext/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettexinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.tex)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/tex"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagemovieannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/movie"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexportfieldsfrompdftoxfdfinstorage(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/export/xfdf"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexportfieldsfrompdftofdfinstorage(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/export/fdf"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpclinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.pcl)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/pcl"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getredactionannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/redaction/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentscreenannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/screen"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetohtml(name: str, specialfolderforsvgimages: str=None, antialiasingprocessing: str='NoAdditionalProcessing', additionalmarginwidthinpoints: int=None, partsembeddingmode: str='EmbedAllIntoHtml', fontsavingmode: str='AlwaysSaveAsWOFF', convertmarkedcontenttolayers: bool=None, htmlmarkupgenerationmode: str='WriteAllHtml', letterspositioningmethod: str='UseEmUnitsAndCompensationOfRoundingErrorsInCss', fontencodingstrategy: str='Default', removeemptyareasontopandbottom: bool=None, documenttype: str='Xhtml', cssclassnamesprefix: str=None, flowlayoutparagraphfullwidth: bool=None, specialfolderforallimages: str=None, folder: str=None, imageresolution: int=None, preventglyphsgrouping: bool=None, pagesflowtypedependsonviewersscreensize: bool=None, minimallinewidth: int=None, explicitlistofsavedpages: str='[]', usezorder: bool=None, defaultfontname: str=None, compresssvggraphicsifany: bool=None, rasterimagessavingmode: str='AsPngImagesEmbeddedIntoSvg', fixedlayout: bool=None, splitcssintopages: bool=None, saveshadowedtextsastransparenttexts: bool=None, savetransparenttexts: bool=None, splitintopages: bool=None, trysavetextunderliningandstrikeoutingincss: bool=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        specialfolderforsvgimages: The path to directory to which must be saved only SVG-images if they are encountered during saving of document as HTML. If parameter is empty or null then SVG files(if any) wil be saved together with other image-files (near to output file) or in special folder for images (if it specified in SpecialImagesFolderIfAny option). It does not affect anything if CustomImageSavingStrategy property was successfully used to process relevant image file.
        antialiasingprocessing: The parameter defines required antialiasing measures during conversion of compound background images from PDF to HTML.
        additionalmarginwidthinpoints: Defines width of margin that will be forcibly left around that output HTML-areas.
        partsembeddingmode: It defines whether referenced files (HTML, Fonts,Images, CSSes) will be embedded into main HTML file or will be generated as apart binary entities.
        fontsavingmode: Defines font saving mode that will be used during saving of PDF to desirable format.
        convertmarkedcontenttolayers: If attribute ConvertMarkedContentToLayers set to true then an all elements inside a PDF marked content (layer) will be put into an HTML div with "data-pdflayer" attribute specifying a layer name. This layer name will be extracted from optional properties of PDF marked content. If this attribute is false (by default) then no any layers will be created from PDF marked content.
        htmlmarkupgenerationmode: Sometimes specific reqirments to generation of HTML markup are present. This parameter defines HTML preparing modes that can be used during conversion of PDF to HTML to match such specific requirments.
        letterspositioningmethod: The mode of positioning of letters in words in result HTML.
        fontencodingstrategy: Defines encoding special rule to tune PDF decoding for current document.
        removeemptyareasontopandbottom: Defines whether in created HTML will be removed top and bottom empty area without any content (if any).
        documenttype: Result document type.
        cssclassnamesprefix: When PDFtoHTML converter generates result CSSs, CSS class names (something like ".stl_01 {}" ... ".stl_NN {}) are generated and used in result CSS. This property allows forcibly set class name prefix.
        flowlayoutparagraphfullwidth: This attribute specifies full width paragraph text for Flow mode, FixedLayout = false.
        specialfolderforallimages: The path to directory to which must be saved any images if they are encountered during saving of document as HTML. If parameter is empty or null then image files(if any) wil be saved together with other files linked to HTML It does not affect anything if CustomImageSavingStrategy property was successfully used to process relevant image file.
        folder: The document folder.
        imageresolution: Resolution for image rendering.
        preventglyphsgrouping: This attribute switch on the mode when text glyphs will not be grouped into words and strings This mode allows to keep maximum precision during positioning of glyphs on the page and it can be used for conversion documents with music notes or glyphs that should be placed separately each other. This parameter will be applied to document only when the value of FixedLayout attribute is true.
        pagesflowtypedependsonviewersscreensize: If attribute 'SplitOnPages=false', than whole HTML representing all input PDF pages will be put into one big result HTML file. This flag defines whether result HTML will be generated in such way that flow of areas that represent PDF pages in result HTML will depend on screen resolution of viewer.
        minimallinewidth: This attribute sets minimal width of graphic path line. If thickness of line is less than 1px Adobe Acrobat rounds it to this value. So this attribute can be used to emulate this behavior for HTML browsers.
        explicitlistofsavedpages: With this property You can explicitely define what pages of document should be converted. Pages in this list must have 1-based numbers. I.e. valid numbers of pages must be taken from range (1...[NumberOfPagesInConvertedDocument]) Order of appearing of pages in this list does not affect their order in result HTML page(s) - in result pages allways will go in order in which they are present in source PDF.
        usezorder: If attribute UseZORder set to true, graphics and text are added to resultant HTML document accordingly Z-order in original PDF document. If this attribute is false all graphics is put as single layer which may cause some unnecessary effects for overlapped objects.
        defaultfontname: Specifies the name of an installed font which is used to substitute any document font that is not embedded and not installed in the system. If null then default substitution font is used.
        compresssvggraphicsifany: The flag that indicates whether found SVG graphics(if any) will be compressed(zipped) into SVGZ format during saving.
        rasterimagessavingmode: Converted PDF can contain raster images This parameter defines how they should be handled during conversion of PDF to HTML.
        fixedlayout: The value indicating whether that HTML is created as fixed layout.
        splitcssintopages: When multipage-mode selected(i.e 'SplitIntoPages' is 'true'), then this attribute defines whether should be created separate CSS-file for each result HTML page.
        saveshadowedtextsastransparenttexts: Pdf can contain texts that are shadowed by another elements (f.e. by images) but can be selected to clipboard in Acrobat Reader (usually it happen when document contains images and OCRed texts extracted from it). This settings tells to converter whether we need save such texts as transparent selectable texts in result HTML to mimic behaviour of Acrobat Reader (othervise such texts are usually saved as hidden, not available for copying to clipboard).
        savetransparenttexts: Pdf can contain transparent texts that can be selected to clipboard (usually it happen when document contains images and OCRed texts extracted from it). This settings tells to converter whether we need save such texts as transparent selectable texts in result HTML.
        splitintopages: The flag that indicates whether each page of source document will be converted into it's own target HTML document, i.e whether result HTML will be splitted into several HTML-pages.
        trysavetextunderliningandstrikeoutingincss: PDF itself does not contain underlining markers for texts. It emulated with line situated under text. This option allows converter try guess that this or that line is a text's underlining and put this info into CSS instead of drawing of underlining graphically.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/html"
    querystring = {}
    if specialfolderforsvgimages:
        querystring['specialFolderForSvgImages'] = specialfolderforsvgimages
    if antialiasingprocessing:
        querystring['antialiasingProcessing'] = antialiasingprocessing
    if additionalmarginwidthinpoints:
        querystring['additionalMarginWidthInPoints'] = additionalmarginwidthinpoints
    if partsembeddingmode:
        querystring['partsEmbeddingMode'] = partsembeddingmode
    if fontsavingmode:
        querystring['fontSavingMode'] = fontsavingmode
    if convertmarkedcontenttolayers:
        querystring['convertMarkedContentToLayers'] = convertmarkedcontenttolayers
    if htmlmarkupgenerationmode:
        querystring['htmlMarkupGenerationMode'] = htmlmarkupgenerationmode
    if letterspositioningmethod:
        querystring['lettersPositioningMethod'] = letterspositioningmethod
    if fontencodingstrategy:
        querystring['fontEncodingStrategy'] = fontencodingstrategy
    if removeemptyareasontopandbottom:
        querystring['removeEmptyAreasOnTopAndBottom'] = removeemptyareasontopandbottom
    if documenttype:
        querystring['documentType'] = documenttype
    if cssclassnamesprefix:
        querystring['cssClassNamesPrefix'] = cssclassnamesprefix
    if flowlayoutparagraphfullwidth:
        querystring['flowLayoutParagraphFullWidth'] = flowlayoutparagraphfullwidth
    if specialfolderforallimages:
        querystring['specialFolderForAllImages'] = specialfolderforallimages
    if folder:
        querystring['folder'] = folder
    if imageresolution:
        querystring['imageResolution'] = imageresolution
    if preventglyphsgrouping:
        querystring['preventGlyphsGrouping'] = preventglyphsgrouping
    if pagesflowtypedependsonviewersscreensize:
        querystring['pagesFlowTypeDependsOnViewersScreenSize'] = pagesflowtypedependsonviewersscreensize
    if minimallinewidth:
        querystring['minimalLineWidth'] = minimallinewidth
    if explicitlistofsavedpages:
        querystring['explicitListOfSavedPages'] = explicitlistofsavedpages
    if usezorder:
        querystring['useZOrder'] = usezorder
    if defaultfontname:
        querystring['defaultFontName'] = defaultfontname
    if compresssvggraphicsifany:
        querystring['compressSvgGraphicsIfAny'] = compresssvggraphicsifany
    if rasterimagessavingmode:
        querystring['rasterImagesSavingMode'] = rasterimagessavingmode
    if fixedlayout:
        querystring['fixedLayout'] = fixedlayout
    if splitcssintopages:
        querystring['splitCssIntoPages'] = splitcssintopages
    if saveshadowedtextsastransparenttexts:
        querystring['saveShadowedTextsAsTransparentTexts'] = saveshadowedtextsastransparenttexts
    if savetransparenttexts:
        querystring['saveTransparentTexts'] = savetransparenttexts
    if splitintopages:
        querystring['splitIntoPages'] = splitintopages
    if trysavetextunderliningandstrikeoutingincss:
        querystring['trySaveTextUnderliningAndStrikeoutingInCss'] = trysavetextunderliningandstrikeoutingincss
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/file/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentpolylineannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/polyline"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagelineannotations(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/line"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetoxml(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/xml"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentlineannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/line"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagestamps(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/stamps"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagetextannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/text"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentattachmentbyindex(name: str, attachmentindex: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        attachmentindex: The attachment index.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/attachments/{attachmentindex}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentsoundannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/sound"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagecomboboxfields(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/combobox"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentsignaturefields(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/signature"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenttextboxfields(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/textbox"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getxmlinstoragetopdf(srcpath: str, storage: str=None, xslfilepath: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.xml)
        storage: The document storage.
        xslfilepath: Full XSL source filename (ex. /folder1/folder2/template.xsl)
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/xml"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    if xslfilepath:
        querystring['xslFilePath'] = xslfilepath
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentpolygonannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/polygon"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagestampannotations(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/stamp"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentbookmarks(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/bookmarks/tree"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagecheckboxfields(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/checkbox"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcomboboxfield(name: str, fieldname: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fieldname: The field name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/combobox/{fieldname}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfreetextannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/freetext"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentradiobuttonfields(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/radiobutton"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimportfieldsfromxmlinstorage(xmlfilepath: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    xmlfilepath: Full source filename (ex. /folder1/folder2/template.xml)
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/import/xml"
    querystring = {'xmlFilePath': xmlfilepath, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsignaturefield(name: str, fieldname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fieldname: The field name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/signature/{fieldname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimportfieldsfromxfdfinstorage(name: str, xfdffilepath: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        xfdffilepath: The XFDF file path.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/import/xfdf"
    querystring = {'xfdfFilePath': xfdffilepath, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentcircleannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/circle"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenttextannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/text"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentsquigglyannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/squiggly"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagepolylineannotations(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/polyline"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscreenannotationdata(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/screen/{annotationid}/data"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetomobixml(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/mobixml"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageradiobuttonfields(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/radiobutton"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcircleannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/circle/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagefreetextannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/freetext"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimages(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/images"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimage(imageid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: Image ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/images/{imageid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstampannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/stamp/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperties(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/documentproperties"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentstamps(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/stamps"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageextractastiff(imageid: str, name: str, folder: str=None, width: int=0, height: int=0, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: Image ID.
        name: The document name.
        folder: The document folder.
        width: The converted image width.
        height: The converted image height.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/images/{imageid}/extract/tiff"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettextannotation(name: str, annotationid: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/text/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmhtinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.mht)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/mht"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetodoc(name: str, mode: str='Textbox', recognizebullets: bool=None, imageresolutionx: int=None, relativehorizontalproximity: int=None, addreturntolineend: bool=None, folder: str=None, imageresolutiony: int=None, maxdistancebetweentextlines: int=None, storage: str=None, format: str='Doc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        mode: Allows to control how a PDF document is converted into a word processing document.
        recognizebullets: Recognize bullets.
        imageresolutionx: Image resolution X.
        relativehorizontalproximity: Relative horizontal proximity.
        addreturntolineend: Add return to line end.
        folder: The document folder.
        imageresolutiony: Image resolution Y.
        maxdistancebetweentextlines: Max distance between text lines.
        storage: The document storage.
        format: Allows to specify .doc or .docx file format.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/doc"
    querystring = {}
    if mode:
        querystring['mode'] = mode
    if recognizebullets:
        querystring['recognizeBullets'] = recognizebullets
    if imageresolutionx:
        querystring['imageResolutionX'] = imageresolutionx
    if relativehorizontalproximity:
        querystring['relativeHorizontalProximity'] = relativehorizontalproximity
    if addreturntolineend:
        querystring['addReturnToLineEnd'] = addreturntolineend
    if folder:
        querystring['folder'] = folder
    if imageresolutiony:
        querystring['imageResolutionY'] = imageresolutiony
    if maxdistancebetweentextlines:
        querystring['maxDistanceBetweenTextLines'] = maxdistancebetweentextlines
    if storage:
        querystring['storage'] = storage
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfield(name: str, fieldname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fieldname: The field name (name should be encoded).
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/{fieldname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettextboxfield(fieldname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    fieldname: The field name.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/textbox/{fieldname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpage(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentcaretannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/caret"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentlistboxfields(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/listbox"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagelinkannotation(linkid: str, pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    linkid: The link ID.
        pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/links/{linkid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfields(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpsinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.ps)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/ps"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetoxps(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/xps"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistboxfield(name: str, fieldname: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fieldname: The field name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/listbox/{fieldname}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentcomboboxfields(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/combobox"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlineannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/line/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageextractasgif(imageid: str, name: str, width: int=0, storage: str=None, height: int=0, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: Image ID.
        name: The document name.
        width: The converted image width.
        storage: The document storage.
        height: The converted image height.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/images/{imageid}/extract/gif"
    querystring = {}
    if width:
        querystring['width'] = width
    if storage:
        querystring['storage'] = storage
    if height:
        querystring['height'] = height
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagecaretannotations(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/caret"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsoundannotationdata(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/sound/{annotationid}/data"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcaretannotation(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/caret/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagetextboxfields(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/textbox"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimportfieldsfromfdfinstorage(name: str, fdffilepath: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fdffilepath: The Fdf file path.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/import/fdf"
    querystring = {'fdfFilePath': fdffilepath, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagesquareannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/square"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getradiobuttonfield(fieldname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    fieldname: The field name.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/radiobutton/{fieldname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethighlightannotation(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/highlight/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsquareannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/square/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getxpsinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.xps)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/xps"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagesignaturefields(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/signature"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttoemf(name: str, pagenumber: int, width: int=0, height: int=0, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        width: The converted image width.
        height: The converted image height.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/emf"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileattachmentannotation(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/fileattachment/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocument(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttobmp(pagenumber: int, name: str, folder: str=None, height: int=0, width: int=0, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        height: The converted image height.
        width: The converted image width.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/bmp"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagescreenannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/screen"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetopptx(name: str, folder: str=None, storage: str=None, slidesasimages: bool=None, separateimages: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        slidesasimages: Slides as images.
        separateimages: Separate images.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/pptx"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if slidesasimages:
        querystring['slidesAsImages'] = slidesasimages
    if separateimages:
        querystring['separateImages'] = separateimages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentpopupannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/popup"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethtmlinstoragetopdf(srcpath: str, marginright: int=None, storage: str=None, islandscape: bool=None, marginleft: int=None, height: int=None, margintop: int=None, width: int=None, marginbottom: int=None, htmlfilename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.zip)
        marginright: Page margin right
        storage: The document storage.
        islandscape: Is page landscaped
        marginleft: Page margin left
        height: Page height
        margintop: Page margin top
        width: Page width
        marginbottom: Page margin bottom
        htmlfilename: Name of HTML file in ZIP.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/html"
    querystring = {'srcPath': srcpath, }
    if marginright:
        querystring['marginRight'] = marginright
    if storage:
        querystring['storage'] = storage
    if islandscape:
        querystring['isLandscape'] = islandscape
    if marginleft:
        querystring['marginLeft'] = marginleft
    if height:
        querystring['height'] = height
    if margintop:
        querystring['marginTop'] = margintop
    if width:
        querystring['width'] = width
    if marginbottom:
        querystring['marginBottom'] = marginbottom
    if htmlfilename:
        querystring['htmlFileName'] = htmlfilename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageunderlineannotations(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/underline"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentredactionannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/redaction"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperty(name: str, propertyname: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/documentproperties/{propertyname}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getverifysignature(name: str, signname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        signname: Sign name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/verifySignature"
    querystring = {'signName': signname, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenttables(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/tables"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentmovieannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/movie"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumenthighlightannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/highlight"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagepopupannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/popup"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getunderlineannotation(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/underline/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentsquareannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/square"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfileattachmentannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/fileattachment"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getepubinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.epub)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/epub"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagetext(lly: int, name: str, urx: int, ury: int, pagenumber: int, llx: int, folder: str=None, format: str='[]', storage: str=None, regex: str=None, splitrects: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    lly: Y - coordinate of lower-left corner.
        name: The document name.
        urx: X - coordinate of upper-right corner.
        ury: Y - coordinate of upper-right corner.
        pagenumber: Number of page (starting from 1).
        llx: X-coordinate of lower - left corner.
        folder: The document folder.
        format: List of formats for search.
        storage: The document storage.
        regex: Formats are specified as a regular expression.
        splitrects: Split result fragments (default is true).
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/text"
    querystring = {'LLY': lly, 'URX': urx, 'URY': ury, 'LLX': llx, }
    if folder:
        querystring['folder'] = folder
    if format:
        querystring['format'] = format
    if storage:
        querystring['storage'] = storage
    if regex:
        querystring['regex'] = regex
    if splitrects:
        querystring['splitRects'] = splitrects
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagecircleannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/circle"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatexinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.tex)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/latex"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttopng(pagenumber: int, name: str, width: int=0, folder: str=None, height: int=0, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        width: The converted image width.
        folder: The document folder.
        height: The converted image height.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/png"
    querystring = {}
    if width:
        querystring['width'] = width
    if folder:
        querystring['folder'] = folder
    if height:
        querystring['height'] = height
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpolylineannotation(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/polyline/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagepolygonannotations(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/polygon"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagesquigglyannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/squiggly"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagelinkannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/links"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmovieannotation(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/movie/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetosvg(name: str, storage: str=None, folder: str=None, compressoutputtoziparchive: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        compressoutputtoziparchive: Specifies whether output will be created as one zip-archive.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/svg"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if compressoutputtoziparchive:
        querystring['compressOutputToZipArchive'] = compressoutputtoziparchive
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageextractasjpeg(imageid: str, name: str, width: int=0, storage: str=None, height: int=0, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: Image ID.
        name: The document name.
        width: The converted image width.
        storage: The document storage.
        height: The converted image height.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/images/{imageid}/extract/jpeg"
    querystring = {}
    if width:
        querystring['width'] = width
    if storage:
        querystring['storage'] = storage
    if height:
        querystring['height'] = height
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstampannotationdata(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/stamp/{annotationid}/data"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentstrikeoutannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/strikeout"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageredactionannotations(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/redaction"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpopupannotation(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/popup/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getxfapdfinstoragetoacroform(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/xfatoacroform"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getapiinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagehighlightannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/highlight"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookmark(bookmarkpath: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    bookmarkpath: The bookmark path.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/bookmarks/bookmark/{bookmarkpath}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getxslfoinstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.xslfo)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/xslfo"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwordsperpage(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/wordCount"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
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
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/storage/exist/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagefileattachmentannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/fileattachment"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdownloaddocumentattachmentbyindex(attachmentindex: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    attachmentindex: The attachment index.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/attachments/{attachmentindex}/download"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileattachmentannotationdata(name: str, annotationid: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/fileattachment/{annotationid}/data"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentattachments(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/attachments"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getexportfieldsfrompdftoxmlinstorage(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/export/xml"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettext(ury: int, urx: int, name: str, lly: int, llx: int, folder: str=None, storage: str=None, regex: str=None, format: str='[]', splitrects: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ury: Y - coordinate of upper-right corner.
        urx: X - coordinate of upper-right corner.
        name: The document name.
        lly: Y - coordinate of lower-left corner.
        llx: X-coordinate of lower - left corner.
        folder: The document folder.
        storage: The document storage.
        regex: Formats are specified as a regular expression.
        format: List of formats for search.
        splitrects: Split result fragments (default is true).
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/text"
    querystring = {'URY': ury, 'URX': urx, 'LLY': lly, 'LLX': llx, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if regex:
        querystring['regex'] = regex
    if format:
        querystring['format'] = format
    if splitrects:
        querystring['splitRects'] = splitrects
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarkdowninstoragetopdf(srcpath: str, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.md)
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/markdown"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscreenannotation(name: str, annotationid: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/screen/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetoxls(name: str, scalefactor: int=None, insertblankcolumnatfirst: bool=None, folder: str=None, storage: str=None, uniformworksheets: bool=None, minimizethenumberofworksheets: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        scalefactor: Scale factor
        insertblankcolumnatfirst: Insert blank column at first
        folder: The document folder.
        storage: The document storage.
        uniformworksheets: Uniform worksheets
        minimizethenumberofworksheets: Minimize the number of worksheets
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/xls"
    querystring = {}
    if scalefactor:
        querystring['scaleFactor'] = scalefactor
    if insertblankcolumnatfirst:
        querystring['insertBlankColumnAtFirst'] = insertblankcolumnatfirst
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if uniformworksheets:
        querystring['uniformWorksheets'] = uniformworksheets
    if minimizethenumberofworksheets:
        querystring['minimizeTheNumberOfWorksheets'] = minimizethenumberofworksheets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetotiff(name: str, folder: str=None, yresolution: int=None, brightness: int=None, compression: str='LZW', height: int=None, storage: str=None, pagecount: int=None, pageindex: int=None, topmargin: int=None, leftmargin: int=None, rightmargin: int=None, xresolution: int=None, colordepth: str='Default', skipblankpages: bool=None, bottommargin: int=None, orientation: str='None', width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        yresolution: Vertical resolution.
        brightness: Image brightness.
        compression: Tiff compression. Possible values are: LZW, CCITT4, CCITT3, RLE, None.
        height: Image height.
        storage: The document storage.
        pagecount: Number of pages to export.
        pageindex: Start page to export.
        topmargin: Top image margin.
        leftmargin: Left image margin.
        rightmargin: Right image margin.
        xresolution: Horizontal resolution.
        colordepth: Image color depth. Possible valuse are: Default, Format8bpp, Format4bpp, Format1bpp.
        skipblankpages: Skip blank pages flag.
        bottommargin: Bottom image margin.
        orientation: Image orientation. Possible values are: None, Landscape, Portait.
        width: Image width.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/tiff"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if yresolution:
        querystring['yResolution'] = yresolution
    if brightness:
        querystring['brightness'] = brightness
    if compression:
        querystring['compression'] = compression
    if height:
        querystring['height'] = height
    if storage:
        querystring['storage'] = storage
    if pagecount:
        querystring['pageCount'] = pagecount
    if pageindex:
        querystring['pageIndex'] = pageindex
    if topmargin:
        querystring['topMargin'] = topmargin
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if xresolution:
        querystring['xResolution'] = xresolution
    if colordepth:
        querystring['colorDepth'] = colordepth
    if skipblankpages:
        querystring['skipBlankPages'] = skipblankpages
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if orientation:
        querystring['orientation'] = orientation
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentstampannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/stamp"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinkannotation(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/ink/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcheckboxfield(name: str, fieldname: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        fieldname: The field name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/checkbox/{fieldname}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentinkannotations(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/ink"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetopdfa(type: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    type: Type of PdfA format.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/pdfa"
    querystring = {'type': type, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsvginstoragetopdf(srcpath: str, margintop: int=None, height: int=None, marginbottom: int=None, width: int=None, storage: str=None, marginright: int=None, islandscape: bool=None, adjustpagesize: bool=None, marginleft: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.svg)
        margintop: Page margin top
        height: Page height
        marginbottom: Page margin bottom
        width: Page width
        storage: The document storage.
        marginright: Page margin right
        islandscape: Is page landscaped
        adjustpagesize: Adjust page size
        marginleft: Page margin left
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/svg"
    querystring = {'srcPath': srcpath, }
    if margintop:
        querystring['marginTop'] = margintop
    if height:
        querystring['height'] = height
    if marginbottom:
        querystring['marginBottom'] = marginbottom
    if width:
        querystring['width'] = width
    if storage:
        querystring['storage'] = storage
    if marginright:
        querystring['marginRight'] = marginright
    if islandscape:
        querystring['isLandscape'] = islandscape
    if adjustpagesize:
        querystring['adjustPageSize'] = adjustpagesize
    if marginleft:
        querystring['marginLeft'] = marginleft
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetolatex(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/latex"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpolygonannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/polygon/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookmarks(bookmarkpath: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    bookmarkpath: The bookmark path.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/bookmarks/list/{bookmarkpath}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetoxlsx(name: str, uniformworksheets: bool=None, scalefactor: int=None, insertblankcolumnatfirst: bool=None, folder: str=None, minimizethenumberofworksheets: bool=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        uniformworksheets: Uniform worksheets
        scalefactor: Scale factor
        insertblankcolumnatfirst: Insert blank column at first
        folder: The document folder.
        minimizethenumberofworksheets: Minimize the number of worksheets
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/xlsx"
    querystring = {}
    if uniformworksheets:
        querystring['uniformWorksheets'] = uniformworksheets
    if scalefactor:
        querystring['scaleFactor'] = scalefactor
    if insertblankcolumnatfirst:
        querystring['insertBlankColumnAtFirst'] = insertblankcolumnatfirst
    if folder:
        querystring['folder'] = folder
    if minimizethenumberofworksheets:
        querystring['minimizeTheNumberOfWorksheets'] = minimizethenumberofworksheets
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagesoundannotations(pagenumber: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/sound"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfainstoragetopdf(srcpath: str, storage: str=None, dontoptimize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    srcpath: Full source filename (ex. /folder1/folder2/template.pdf)
        storage: The document storage.
        dontoptimize: If set, document resources will not be optimized.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/pdfa"
    querystring = {'srcPath': srcpath, }
    if storage:
        querystring['storage'] = storage
    if dontoptimize:
        querystring['dontOptimize'] = dontoptimize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentdisplayproperties(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/displayproperties"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagestrikeoutannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/strikeout"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetoepub(name: str, storage: str=None, folder: str=None, contentrecognitionmode: str='Flow', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        contentrecognitionmode: Property tunes conversion for this or that desirable method of recognition of content.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/epub"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if contentrecognitionmode:
        querystring['contentRecognitionMode'] = contentrecognitionmode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageinkannotations(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations/ink"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstrikeoutannotation(name: str, annotationid: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        annotationid: The annotation ID.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/strikeout/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagetables(name: str, pagenumber: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/tables"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlinkannotation(name: str, linkid: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        linkid: The link ID.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/links/{linkid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpages(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsquigglyannotation(annotationid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/squiggly/{annotationid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsoundannotation(annotationid: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    annotationid: The annotation ID.
        name: The document name.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/sound/{annotationid}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttotiff(pagenumber: int, name: str, width: int=0, storage: str=None, folder: str=None, height: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        width: The converted image width.
        storage: The document storage.
        folder: The document folder.
        height: The converted image height.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/tiff"
    querystring = {}
    if width:
        querystring['width'] = width
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimageextractaspng(imageid: str, name: str, folder: str=None, width: int=0, height: int=0, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    imageid: Image ID.
        name: The document name.
        folder: The document folder.
        width: The converted image width.
        height: The converted image height.
        storage: The document storage.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/images/{imageid}/extract/png"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentunderlineannotations(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/annotations/underline"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttogif(pagenumber: int, name: str, height: int=0, storage: str=None, width: int=0, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        height: The converted image height.
        storage: The document storage.
        width: The converted image width.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/gif"
    querystring = {}
    if height:
        querystring['height'] = height
    if storage:
        querystring['storage'] = storage
    if width:
        querystring['width'] = width
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageconverttojpeg(pagenumber: int, name: str, storage: str=None, width: int=0, folder: str=None, height: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        width: The converted image width.
        folder: The document folder.
        height: The converted image height.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/convert/jpeg"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if width:
        querystring['width'] = width
    if folder:
        querystring['folder'] = folder
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageannotations(pagenumber: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagenumber: The page number.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/pages/{pagenumber}/annotations"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettable(tableid: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    tableid: The table ID.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/tables/{tableid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpdfinstoragetotex(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/convert/tex"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagelistboxfields(name: str, pagenumber: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        pagenumber: The page number.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/page/{pagenumber}/fields/listbox"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentcheckboxfields(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/{name}/fields/checkbox"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebinstoragetopdf(url: str, storage: str=None, height: int=None, marginright: int=None, marginleft: int=None, margintop: int=None, marginbottom: int=None, islandscape: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: Source url
        storage: The document storage.
        height: Page height
        marginright: Page margin right
        marginleft: Page margin left
        margintop: Page margin top
        marginbottom: Page margin bottom
        islandscape: Is page landscaped
        width: Page width
        
    """
    url = f"https://aspose-pdf-cloud1.p.rapidapi.com/pdf/create/web"
    querystring = {'url': url, }
    if storage:
        querystring['storage'] = storage
    if height:
        querystring['height'] = height
    if marginright:
        querystring['marginRight'] = marginright
    if marginleft:
        querystring['marginLeft'] = marginleft
    if margintop:
        querystring['marginTop'] = margintop
    if marginbottom:
        querystring['marginBottom'] = marginbottom
    if islandscape:
        querystring['isLandscape'] = islandscape
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-pdf-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

