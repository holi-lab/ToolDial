import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_barcode_v2(text: str, mod43: bool=None, linecolor: str='#000000', ean128: bool=None, marginbottom: int=10, width: int=4, flat: bool=None, textposition: str=None, margin: int=10, margintop: int=10, marginleft: int=10, background: str='#ccffff', fontsize: int=20, textmargin: int=2, marginright: int=10, displayvalue: bool=None, textalign: str=None, height: int=40, barcodetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Barcode with the supplied "text" parameter. There are 20 Barcode Types. Refer to the Docs section for details."
    text: The text that will be converted to a barcode.
Default: 12345678910
        mod43: Only applicable for **CODE39** 
CODE39 can be used with an optional Modulo 43 check digit. You can do this by setting the mod43 option to true.  If you want to use this feature, you need to enable it in the barcode reader. 
Default: false
        linecolor: Bar and text color of the barcode in hexadecimal color code.
Default: #000000
        ean128: Only applicable for **CODE128**
CODE128 can be encoded as GS1-128/EAN-128. Set the **ean128** to **true** to use this feature.
Default: false
        marginbottom: Margin on the bottom side of the barcode. Setting this will override the **margin **parameter value.
Default: 10
        width: The width of a single bar.
Default: 4
        flat: Only applicable for **EAN13**/**EAN8**/**UPC**/ **UPCE**
EAN13, EAN8, UPC and UPCE barcodes can be rendered without  guard bars. If you want a flat rendering, you can specify the flat option and skip the guard bars.
Default: false
        textposition: Vertical position of the text value. Select from top/bottom.
Default: bottom
        margin: Set the space margin around the barcode. If nothing else is set, all sides will inherit this margin property, but they can also be set separately.
Default: 10
        margintop: Margin on the top side of the barcode. Setting this will override the **margin **parameter value.
Default: 10
        marginleft: Margin on the left side of the barcode. Setting this will override the **margin **parameter value.
Default: 10
        background: Background color of the barcode in hexadecimal color code.
Default: #ccffff
        fontsize: Font size of the text value.
Default: 20
        textmargin: The gap between text and barcode.
Default: 2
        marginright: Margin on the right side of the barcode. Setting this will override the **margin **parameter value.
Default: 10
        displayvalue: Show/hide the text of the barcode.
Default: true.
        textalign: Horizontal alignment of the text.  Select from left/center/right.
Default: center
        height: The height of the barcode.
Default: 40
        barcodetype: Select one from the following barcode type: CODE39, CODE128, CODE128A, CODE128B, CODE128C, EAN13, EAN8, EAN5, EAN2, UPC, UPCE, ITF14, ITF, MSI, MSI10, MSI11, MSI1010, MSI1110, pharmacode, codabar.
Default: **CODE128**
        
    """
    url = f"https://barcode-generator4.p.rapidapi.com/"
    querystring = {'text': text, }
    if mod43:
        querystring['mod43'] = mod43
    if linecolor:
        querystring['lineColor'] = linecolor
    if ean128:
        querystring['ean128'] = ean128
    if marginbottom:
        querystring['marginBottom'] = marginbottom
    if width:
        querystring['width'] = width
    if flat:
        querystring['flat'] = flat
    if textposition:
        querystring['textPosition'] = textposition
    if margin:
        querystring['margin'] = margin
    if margintop:
        querystring['marginTop'] = margintop
    if marginleft:
        querystring['marginLeft'] = marginleft
    if background:
        querystring['background'] = background
    if fontsize:
        querystring['fontSize'] = fontsize
    if textmargin:
        querystring['textMargin'] = textmargin
    if marginright:
        querystring['marginRight'] = marginright
    if displayvalue:
        querystring['displayValue'] = displayvalue
    if textalign:
        querystring['textAlign'] = textalign
    if height:
        querystring['height'] = height
    if barcodetype:
        querystring['barcodeType'] = barcodetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-generator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

