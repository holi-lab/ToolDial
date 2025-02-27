import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def label_templates(format: str, brand: str='avery', height: str=None, code: str='2x2', width: str=None, shape: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of label templates filtered by one or more of the following parameters:
		paper format
		template brand
		exact label size
		approximate label size
		label shape"
    format: Its value can be either 
**Letter**
or 
**A4** 
They represent the two world standard printer paper sizes, respectively 8.5 inch by 11 inch and 210 mm by 297 mm.
        brand: Label manufacturer's name.
This is the current list of possible values. We will continue adding new brands.
Avery
Herma
Onlinelabels
Uline
Megastar
Sheetlabels
        height: Label height.
        code: It is a string that represents **average** label sizes, in the format [width]x[height].
For example:
**2.5x4**
or 
**70x40**

Being average sizes, means that they represent all sizes that are close to them. 
E.g. **3x2** would represent any other close sizes like **3.062x1.837**.

This concept is useful, when you are going to suggest to your users, which templates do offer label sizes that are equal of close to the label size they need to print on.

        width: Label width
        shape: Label shape.
Valid values are:
square
rectangle
circle
oval
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labels/{format}"
    querystring = {}
    if brand:
        querystring['brand'] = brand
    if height:
        querystring['height'] = height
    if code:
        querystring['code'] = code
    if width:
        querystring['width'] = width
    if shape:
        querystring['shape'] = shape
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def print_on_dynamically_generated_label_templates(rightmargin: int, labelheight: int, number: int, bottommargin: int, topmargin: int, templatecode: str, leftmargin: int, labelwidth: int, labelcontent: str, heightratio: int=None, marker: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the PDF file which contains the printout of labels.
		
		The label template will be generated on the fly, based on the label layout dimensions that you will supply on every request.
		The content printed on each label, is the image whose URL is supplied by **labelContent** query parameter."
    rightmargin: The distance from the right edge of the paper to the right edge of the last column of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelheight: The height of the label.
When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.

        number: The total number of labels.
        bottommargin: The distance from the bottom edge of the paper to the bottom edge of the last row of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        topmargin: The distance from the top edge of the paper to the top edge of the first row of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        templatecode: It is the paper format upon which the system will generate the label layout template.
Valid values are **A4** and **Letter** 
        leftmargin: The distance from the left edge of the paper to the left edge of the first column of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelwidth: The width of the label.
When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelcontent: The Url of the image that you are going to print on labels.
        heightratio: The percentage of label height that will be occupied by the printed image. E.g. value **80** means that there will be a 10% blank space on top, and another 10% at the bottom of each label.

Default value is **80**.
Minimum acceptable value is **40**, and the maximum is **95**.
        marker: Markers help making sure that printing is perfectly centered on each label.
By default no marker will be printed.

Possible values are:
**0** - no markers
**1**-  Dots
**2**- Ccrosses
**3**- Lines
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labels"
    querystring = {'rightMargin': rightmargin, 'labelHeight': labelheight, 'number': number, 'bottomMargin': bottommargin, 'topMargin': topmargin, 'templateCode': templatecode, 'leftMargin': leftmargin, 'labelWidth': labelwidth, 'labelContent': labelcontent, }
    if heightratio:
        querystring['heightRatio'] = heightratio
    if marker:
        querystring['marker'] = marker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def print_on_dynamically_generated_label_templates(rightmargin: int, labelheight: int, number: int, bottommargin: int, topmargin: int, templatecode: str, leftmargin: int, labelwidth: int, labelcontent: str, heightratio: int=None, marker: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the PDF file which contains the printout of labels.
		
		The label template will be generated on the fly, based on the label layout dimensions that you will supply on every request.
		The content printed on each label, is the image whose URL is supplied by **labelContent** query parameter."
    rightmargin: The distance from the right edge of the paper to the right edge of the last column of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelheight: The height of the label.
When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.

        number: The total number of labels.
        bottommargin: The distance from the bottom edge of the paper to the bottom edge of the last row of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        topmargin: The distance from the top edge of the paper to the top edge of the first row of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        templatecode: It is the paper format upon which the system will generate the label layout template.
Valid values are **A4** and **Letter** 
        leftmargin: The distance from the left edge of the paper to the left edge of the first column of labels.

When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelwidth: The width of the label.
When the template code is **Letter**,  its value will count in inches.
When the template code is **A4**,  its value will count in millimeters.
        labelcontent: The Url of the image that you are going to print on labels.
        heightratio: The percentage of label height that will be occupied by the printed image. E.g. value **80** means that there will be a 10% blank space on top, and another 10% at the bottom of each label.

Default value is **80**.
Minimum acceptable value is **40**, and the maximum is **95**.
        marker: Markers help making sure that printing is perfectly centered on each label.
By default no marker will be printed.

Possible values are:
**0** - no markers
**1**-  Dots
**2**- Ccrosses
**3**- Lines
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labels"
    querystring = {'rightMargin': rightmargin, 'labelHeight': labelheight, 'number': number, 'bottomMargin': bottommargin, 'topMargin': topmargin, 'templateCode': templatecode, 'leftMargin': leftmargin, 'labelWidth': labelwidth, 'labelContent': labelcontent, }
    if heightratio:
        querystring['heightRatio'] = heightratio
    if marker:
        querystring['marker'] = marker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def print_on_label_templates_or_plain_paper(templatecode: str, labelcontent: str, verticaloffset: int=None, marker: int=None, heightratio: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the PDF file which contains the printout of labels.
		
		Label layout conforms the label template and three print setup query parameters.
		The content printed on each label, is the image whose URL is supplied by **labelContent** query parameter."
    templatecode: Label sheet manufactures, assign a unique code to each of their blank label templates. 
        labelcontent: The Url of the image that you are going to print on labels.
        verticaloffset: Some printers can feed the paper a bit in advance or delay. As a result, the printout is a bit off by 1 or 2 millimeters in the vertical direction, and the content of the labels is not centered.
This parameter helps you fix this problem and come out with a perfectly centered printout.
E.g. if the printout is positioned about 2mm lower than where it should, set this value to 2 and try again.
If the printout is positioned higher, set a negative value.
Play with this value until your printout is perfectly centered on all labels.

The default values is zero.
Acceptable values are: -3, -2, -1, 0, 1, 2 and 3.

        marker: Markers help making sure that printing is perfectly centered on each label.
By default no marker will be printed.

Possible values are:
**0** - no markers
**1**-  Dots
**2**- Ccrosses
**3**- Lines
        heightratio: The percentage of label height that will be occupied by the printed image. E.g. value **80** means that there will be a 10% blank space on top, and another 10% at the bottom of each label.

Default value is **80**.
Minimum acceptable value is **40**, and the maximum is **95**.
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labels"
    querystring = {'templateCode': templatecode, 'labelContent': labelcontent, }
    if verticaloffset:
        querystring['verticalOffset'] = verticaloffset
    if marker:
        querystring['marker'] = marker
    if heightratio:
        querystring['heightRatio'] = heightratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def print_on_label_templates_or_plain_paper(templatecode: str, labelcontent: str, verticaloffset: int=None, marker: int=None, heightratio: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the PDF file which contains the printout of labels.
		
		Label layout conforms the label template and three print setup query parameters.
		The content printed on each label, is the image whose URL is supplied by **labelContent** query parameter."
    templatecode: Label sheet manufactures, assign a unique code to each of their blank label templates. 
        labelcontent: The Url of the image that you are going to print on labels.
        verticaloffset: Some printers can feed the paper a bit in advance or delay. As a result, the printout is a bit off by 1 or 2 millimeters in the vertical direction, and the content of the labels is not centered.
This parameter helps you fix this problem and come out with a perfectly centered printout.
E.g. if the printout is positioned about 2mm lower than where it should, set this value to 2 and try again.
If the printout is positioned higher, set a negative value.
Play with this value until your printout is perfectly centered on all labels.

The default values is zero.
Acceptable values are: -3, -2, -1, 0, 1, 2 and 3.

        marker: Markers help making sure that printing is perfectly centered on each label.
By default no marker will be printed.

Possible values are:
**0** - no markers
**1**-  Dots
**2**- Ccrosses
**3**- Lines
        heightratio: The percentage of label height that will be occupied by the printed image. E.g. value **80** means that there will be a 10% blank space on top, and another 10% at the bottom of each label.

Default value is **80**.
Minimum acceptable value is **40**, and the maximum is **95**.
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labels"
    querystring = {'templateCode': templatecode, 'labelContent': labelcontent, }
    if verticaloffset:
        querystring['verticalOffset'] = verticaloffset
    if marker:
        querystring['marker'] = marker
    if heightratio:
        querystring['heightRatio'] = heightratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def label_shapes(format: str, brand: str='Megastar', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of available label shapes."
    format: Its value can be either 
**Letter**
or 
**A4** 
They represent the two world standard printer paper sizes, respectively 8.5 inch by 11 inch and 210 mm by 297 mm.
        brand: Label manufacturer's name.
This is the current list of possible values. We will continue adding new brands.
Avery
Herma
Onlinelabels
Uline
Megastar
Sheetlabels
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labelShapes/{format}"
    querystring = {}
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def label_sizes(format: str, shape: str='square', brand: str='avery', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of label sizes in the format [width]x[height].
		
		These sizes are rounded, meaning that they can represent a few other sizes that are close to them. For example size 4"x3", will represent 3.75"x3",  4"x3.25" and so on."
    format: Its value can be either 
**Letter**
or 
**A4** 
They represent the two world standard printer paper sizes, respectively 8.5 inch by 11 inch and 210 mm by 297 mm.
        brand: Label manufacturer's name.
This is the current list of possible values. We will continue adding new brands.
Avery
Herma
Onlinelabels
Uline
Megastar
Sheetlabels
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labelSizes/{format}"
    querystring = {}
    if shape:
        querystring['shape'] = shape
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def label_template_brands(format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of blank label sheet brands that Ecardify supports to date. If your preferred brand is missing, please let us know."
    format: Its value can be either 
**Letter**
or 
**A4** 
They represent the two world standard printer paper sizes, respectively 8.5 inch by 11 inch and 210 mm by 297 mm.
        
    """
    url = f"https://print-your-own-labels-and-stickers.p.rapidapi.com/labelBrands/{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "print-your-own-labels-and-stickers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

