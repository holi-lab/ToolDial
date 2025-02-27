import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def objectexists(path: str, versionid: str=None, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        versionid: File version ID
        storagename: Storage name
        
    """
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/exist/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
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
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbarcodegenerate(type: str, text: str, resolution: int=None, imagewidth: int=None, supplementdata: str=None, textlocation: str='Below', fontsizemode: str='Auto', rotationangle: int=None, supplementspace: int=None, imageheight: int=None, enableescape: bool=None, resolutiony: int=None, sizemode: str='None', barheight: int=None, textcolor: str=None, textspace: int=None, widenarrowratio: int=None, barcolor: str=None, dimensionx: int=None, nowrap: bool=None, twoddisplaytext: str=None, borderwidth: int=None, bordercolor: str=None, filledbars: bool=None, bordervisible: bool=None, barwidthreduction: int=None, backcolor: str=None, alwaysshowchecksum: bool=None, format: str=None, borderdashstyle: str='Solid', units: str='Pixel', enablechecksum: str='Default', validatetext: bool=None, resolutionx: int=None, textalignment: str='Left', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    type: Type of barcode to generate.
        text: Text to encode.
        resolution: Resolution of the BarCode image.
One value for both dimensions.
Default value: 96 dpi.
        imagewidth: Width of the barcode image in given units. Default units: pixel.
        supplementdata: Supplement parameters.
Used for Interleaved2of5, Standard2of5, EAN13, EAN8, UPCA, UPCE, ISBN, ISSN, ISMN.
        textlocation: Specify the displaying Text Location, set to CodeLocation.None to hide CodeText.
Default value: CodeLocation.Below.
        fontsizemode: Specify FontSizeMode. If FontSizeMode is set to Auto, font size will be calculated automatically based on xDimension value.
It is recommended to use FontSizeMode.Auto especially in AutoSizeMode.Nearest or AutoSizeMode.Interpolation.
Default value: FontSizeMode.Auto.
        rotationangle: BarCode image rotation angle, measured in degree, e.g. RotationAngle = 0 or RotationAngle = 360 means no rotation.
If RotationAngle NOT equal to 90, 180, 270 or 0, it may increase the difficulty for the scanner to read the image.
Default value: 0.
        supplementspace: Space between main the BarCode and supplement BarCode.
        imageheight: Height of the barcode image in given units. Default units: pixel.
        enableescape: Indicates whether explains the character "\" as an escape character in CodeText property. Used for Pdf417, DataMatrix, Code128 only
If the EnableEscape is true, "\" will be explained as a special escape character. Otherwise, "\" acts as normal characters.
Aspose.BarCode supports input decimal ascii code and mnemonic for ASCII control-code characters. For example, \013 and \\CR stands for CR.
        resolutiony: DEPRECATED: Use 'Resolution' instead.
        sizemode: Specifies the different types of automatic sizing modes.
Default value: AutoSizeMode.None.
        barheight: Height of the barcode in given units. Default units: pixel.
        textcolor: Specify the displaying CodeText's Color.
Default value: Color.Black.
        textspace: Space between the CodeText and the BarCode in Unit value.
Default value: 2pt.
Ignored for EAN8, EAN13, UPCE, UPCA, ISBN, ISMN, ISSN, UpcaGs1DatabarCoupon.
        widenarrowratio: Wide bars to Narrow bars ratio.
Default value: 3, that is, wide bars are 3 times as wide as narrow bars.
Used for ITF, PZN, PharmaCode, Standard2of5, Interleaved2of5, Matrix2of5, ItalianPost25, IATA2of5, VIN, DeutschePost, OPC, Code32, DataLogic2of5, PatchCode, Code39Extended, Code39Standard
        barcolor: Bars color.
Default value: Color.Black.
        dimensionx: The smallest width of the unit of BarCode bars or spaces.
Increase this will increase the whole barcode image width.
Ignored if AutoSizeMode property is set to AutoSizeMode.Nearest or AutoSizeMode.Interpolation.
        nowrap: Specify word wraps (line breaks) within text.
Default value: false.
        twoddisplaytext: Text that will be displayed instead of codetext in 2D barcodes.
Used for: Aztec, Pdf417, DataMatrix, QR, MaxiCode, DotCode
        borderwidth: Border width.
Default value: 0.
Ignored if Visible is set to false.
        bordercolor: Border color.
Default value: Color.Black.
        filledbars: Value indicating whether bars are filled.
Only for 1D barcodes.
Default value: true.
        bordervisible: Border visibility. If false than parameter Width is always ignored (0).
Default value: false.
        barwidthreduction: Bars reduction value that is used to compensate ink spread while printing.
        backcolor: Background color of the barcode image.
Default value: Color.White.
        alwaysshowchecksum: Always display checksum digit in the human readable text for Code128 and GS1Code128 barcodes.
        format: Result image format.
        borderdashstyle: Border dash style.
Default value: BorderDashStyle.Solid.
        units: Common Units for all measuring in query. Default units: pixel.
        enablechecksum: Enable checksum during generation 1D barcodes.
Default is treated as Yes for symbology which must contain checksum, as No where checksum only possible.
Checksum is possible: Code39 Standard/Extended, Standard2of5, Interleaved2of5, Matrix2of5, ItalianPost25, DeutschePostIdentcode, DeutschePostLeitcode, VIN, Codabar
Checksum always used: Rest symbology
        validatetext: Only for 1D barcodes.
If codetext is incorrect and value set to true - exception will be thrown. Otherwise codetext will be corrected to match barcode's specification.
Exception always will be thrown for: Databar symbology if codetext is incorrect.
Exception always will not be thrown for: AustraliaPost, SingaporePost, Code39Extended, Code93Extended, Code16K, Code128 symbology if codetext is incorrect.
        resolutionx: DEPRECATED: Use 'Resolution' instead.
        textalignment: Text alignment.
        
    """
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/generate"
    querystring = {'Type': type, 'Text': text, }
    if resolution:
        querystring['Resolution'] = resolution
    if imagewidth:
        querystring['ImageWidth'] = imagewidth
    if supplementdata:
        querystring['SupplementData'] = supplementdata
    if textlocation:
        querystring['TextLocation'] = textlocation
    if fontsizemode:
        querystring['FontSizeMode'] = fontsizemode
    if rotationangle:
        querystring['RotationAngle'] = rotationangle
    if supplementspace:
        querystring['SupplementSpace'] = supplementspace
    if imageheight:
        querystring['ImageHeight'] = imageheight
    if enableescape:
        querystring['EnableEscape'] = enableescape
    if resolutiony:
        querystring['ResolutionY'] = resolutiony
    if sizemode:
        querystring['SizeMode'] = sizemode
    if barheight:
        querystring['BarHeight'] = barheight
    if textcolor:
        querystring['TextColor'] = textcolor
    if textspace:
        querystring['TextSpace'] = textspace
    if widenarrowratio:
        querystring['WideNarrowRatio'] = widenarrowratio
    if barcolor:
        querystring['BarColor'] = barcolor
    if dimensionx:
        querystring['DimensionX'] = dimensionx
    if nowrap:
        querystring['NoWrap'] = nowrap
    if twoddisplaytext:
        querystring['TwoDDisplayText'] = twoddisplaytext
    if borderwidth:
        querystring['BorderWidth'] = borderwidth
    if bordercolor:
        querystring['BorderColor'] = bordercolor
    if filledbars:
        querystring['FilledBars'] = filledbars
    if bordervisible:
        querystring['BorderVisible'] = bordervisible
    if barwidthreduction:
        querystring['BarWidthReduction'] = barwidthreduction
    if backcolor:
        querystring['BackColor'] = backcolor
    if alwaysshowchecksum:
        querystring['AlwaysShowChecksum'] = alwaysshowchecksum
    if format:
        querystring['format'] = format
    if borderdashstyle:
        querystring['BorderDashStyle'] = borderdashstyle
    if units:
        querystring['Units'] = units
    if enablechecksum:
        querystring['EnableChecksum'] = enablechecksum
    if validatetext:
        querystring['ValidateText'] = validatetext
    if resolutionx:
        querystring['ResolutionX'] = resolutionx
    if textalignment:
        querystring['TextAlignment'] = textalignment
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbarcoderecognize(name: str, allowqrmicroqrrestoration: bool=None, detectencoding: bool=None, australianpostencodingtable: str='CTable', storage: str=None, folder: str=None, allowcomplexbackground: bool=None, rectangleregion: str=None, allowsaltandpepperfiltering: bool=None, readtinybarcodes: bool=None, skipdiagonalsearch: bool=None, rectx: int=None, preset: str='HighPerformance', allowregularimage: bool=None, allowincorrectbarcodes: bool=None, checkmore1dvariants: bool=None, timeout: int=None, scanwindowsizes: str='[]', allowonedfastbarcodesdetector: bool=None, stripfnc: bool=None, regionlikelihoodthresholdpercent: int=None, allowdecreasedimage: bool=None, allowwhitespotsremoving: bool=None, recty: int=None, rectwidth: int=None, allowmediansmoothing: bool=None, allowdatamatrixindustrialbarcodes: bool=None, rectheight: int=None, type: str='all', checksumvalidation: str='Default', similarity: int=None, allowinvertimage: bool=None, allowonedwipedbarsrestoration: bool=None, allowdetectscangap: bool=None, mediansmoothingwindowsize: int=None, allowmicrowhitespotsremoving: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The image file name.
        allowqrmicroqrrestoration: Allows engine for QR/MicroQR to recognize damaged MicroQR barcodes.
        detectencoding: A flag which force engine to detect codetext encoding for Unicode.
        australianpostencodingtable: Interpreting Type for the Customer Information of AustralianPost BarCode.Default is CustomerInformationInterpretingType.Other.
        storage: The image storage.
        folder: The image folder.
        allowcomplexbackground: Allows engine to recognize color barcodes on color background as additional scan. Extremely slow mode.
        allowsaltandpepperfiltering: Allows engine to recognize barcodes with salt and pepper noise type. Mode can remove small noise with white and black dots.
        readtinybarcodes: Allows engine to recognize tiny barcodes on large images. Ignored if AllowIncorrectBarcodes is set to True. Default value: False.
        skipdiagonalsearch: Allows detector to skip search for diagonal barcodes.
Setting it to false will increase detection time but allow to find diagonal barcodes that can be missed otherwise.
Enabling of diagonal search leads to a bigger detection time.
        rectx: Set X for area for recognition.
        preset: Preset allows to configure recognition quality and speed manually.
You can quickly set up Preset by embedded presets: HighPerformance, NormalQuality,
HighQuality, MaxBarCodes or you can manually configure separate options.
Default value of Preset is NormalQuality.
        allowregularimage: Allows engine to recognize regular image without any restorations as main scan. Mode to recognize image as is.
        allowincorrectbarcodes: Allows engine to recognize barcodes which has incorrect checksum or incorrect values.
Mode can be used to recognize damaged barcodes with incorrect text.
        checkmore1dvariants: Allows engine to recognize 1D barcodes with checksum by checking more recognition variants. Default value: False.
        timeout: Timeout of recognition process.
        scanwindowsizes: Scan window sizes in pixels.
Allowed sizes are 10, 15, 20, 25, 30.
Scanning with small window size takes more time and provides more accuracy but may fail in detecting very big barcodes.
Combining of several window sizes can improve detection quality.
        allowonedfastbarcodesdetector: Allows engine for 1D barcodes to quickly recognize high quality barcodes which fill almost whole image.
Mode helps to quickly recognize generated barcodes from Internet.
        stripfnc: Value indicating whether FNC symbol strip must be done.
        regionlikelihoodthresholdpercent: Sets threshold for detected regions that may contain barcodes.
Value 0.7 means that bottom 70% of possible regions are filtered out and not processed further.
Region likelihood threshold must be between [0.05, 0.9]
Use high values for clear images with few barcodes.
Use low values for images with many barcodes or for noisy images.
Low value may lead to a bigger recognition time.
        allowdecreasedimage: Allows engine to recognize decreased image as additional scan. Size for decreasing is selected by internal engine algorithms.
Mode helps to recognize barcodes which are noised and blurred but captured with high resolution.
        allowwhitespotsremoving: Allows engine to recognize image without small white spots as additional scan. Mode helps to recognize noised image as well as median smoothing filtering.
        recty: Set Y for area for recognition.
        rectwidth: Set Width of area for recognition.
        allowmediansmoothing: Allows engine to enable median smoothing as additional scan. Mode helps to recognize noised barcodes.
        allowdatamatrixindustrialbarcodes: Allows engine for Datamatrix to recognize dashed industrial Datamatrix barcodes.
Slow mode which helps only for dashed barcodes which consist from spots.
        rectheight: Set Height of area for recognition.
        type: The type of barcode to read.
        checksumvalidation: Enable checksum validation during recognition for 1D barcodes.
Default is treated as Yes for symbologies which must contain checksum, as No where checksum only possible.
Checksum never used: Codabar
Checksum is possible: Code39 Standard/Extended, Standard2of5, Interleaved2of5, Matrix2of5, ItalianPost25, DeutschePostIdentcode, DeutschePostLeitcode, VIN
Checksum always used: Rest symbologies
        similarity: Similarity coefficient depends on how homogeneous barcodes are.
Use high value for for clear barcodes.
Use low values to detect barcodes that ara partly damaged or not lighten evenly.
Similarity coefficient must be between [0.5, 0.9]
        allowinvertimage: Allows engine to recognize inverse color image as additional scan. Mode can be used when barcode is white on black background.
        allowonedwipedbarsrestoration: Allows engine for 1D barcodes to recognize barcodes with single wiped/glued bars in pattern.
        allowdetectscangap: Allows engine to use gap between scans to increase recognition speed. Mode can make recognition problems with low height barcodes.
        mediansmoothingwindowsize: Window size for median smoothing. Typical values are 3 or 4. Default value is 3. AllowMedianSmoothing must be set.
        allowmicrowhitespotsremoving: Allows engine for Postal barcodes to recognize slightly noised images. Mode helps to recognize slightly damaged Postal barcodes.
        
    """
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/{name}/recognize"
    querystring = {}
    if allowqrmicroqrrestoration:
        querystring['AllowQRMicroQrRestoration'] = allowqrmicroqrrestoration
    if detectencoding:
        querystring['DetectEncoding'] = detectencoding
    if australianpostencodingtable:
        querystring['AustralianPostEncodingTable'] = australianpostencodingtable
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if allowcomplexbackground:
        querystring['AllowComplexBackground'] = allowcomplexbackground
    if rectangleregion:
        querystring['RectangleRegion'] = rectangleregion
    if allowsaltandpepperfiltering:
        querystring['AllowSaltAndPepperFiltering'] = allowsaltandpepperfiltering
    if readtinybarcodes:
        querystring['ReadTinyBarcodes'] = readtinybarcodes
    if skipdiagonalsearch:
        querystring['SkipDiagonalSearch'] = skipdiagonalsearch
    if rectx:
        querystring['RectX'] = rectx
    if preset:
        querystring['Preset'] = preset
    if allowregularimage:
        querystring['AllowRegularImage'] = allowregularimage
    if allowincorrectbarcodes:
        querystring['AllowIncorrectBarcodes'] = allowincorrectbarcodes
    if checkmore1dvariants:
        querystring['CheckMore1DVariants'] = checkmore1dvariants
    if timeout:
        querystring['Timeout'] = timeout
    if scanwindowsizes:
        querystring['ScanWindowSizes'] = scanwindowsizes
    if allowonedfastbarcodesdetector:
        querystring['AllowOneDFastBarcodesDetector'] = allowonedfastbarcodesdetector
    if stripfnc:
        querystring['StripFNC'] = stripfnc
    if regionlikelihoodthresholdpercent:
        querystring['RegionLikelihoodThresholdPercent'] = regionlikelihoodthresholdpercent
    if allowdecreasedimage:
        querystring['AllowDecreasedImage'] = allowdecreasedimage
    if allowwhitespotsremoving:
        querystring['AllowWhiteSpotsRemoving'] = allowwhitespotsremoving
    if recty:
        querystring['RectY'] = recty
    if rectwidth:
        querystring['RectWidth'] = rectwidth
    if allowmediansmoothing:
        querystring['AllowMedianSmoothing'] = allowmediansmoothing
    if allowdatamatrixindustrialbarcodes:
        querystring['AllowDatamatrixIndustrialBarcodes'] = allowdatamatrixindustrialbarcodes
    if rectheight:
        querystring['RectHeight'] = rectheight
    if type:
        querystring['Type'] = type
    if checksumvalidation:
        querystring['ChecksumValidation'] = checksumvalidation
    if similarity:
        querystring['Similarity'] = similarity
    if allowinvertimage:
        querystring['AllowInvertImage'] = allowinvertimage
    if allowonedwipedbarsrestoration:
        querystring['AllowOneDWipedBarsRestoration'] = allowonedwipedbarsrestoration
    if allowdetectscangap:
        querystring['AllowDetectScanGap'] = allowdetectscangap
    if mediansmoothingwindowsize:
        querystring['MedianSmoothingWindowSize'] = mediansmoothingwindowsize
    if allowmicrowhitespotsremoving:
        querystring['AllowMicroWhiteSpotsRemoving'] = allowmicrowhitespotsremoving
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
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
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
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
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
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
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/file/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
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
    url = f"https://aspose-barcode-cloud.p.rapidapi.com/barcode/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-barcode-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

