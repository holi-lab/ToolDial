import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def csv2pdf(orientation: str, url: str, filename: str='invoice.pdf', titlealign: str='left', subtitlealign: str='right', highlightfirstcolumn: bool=None, highlightlastcolumn: bool=None, data: str=None, template: str='relax', separator: str=';', subtitle: str='<b>Billing to : </b><br/>John Customer<br>Branch Street 10<br/>BB 14021 Charmtown<br/><br/><br/><i>Invoice Date : 04/05/2016</i><br>Invoice Nr : <b>2016.00001</b>', title: str='<b>Invoice</b><br/><br/><br/><b>Example Industries </b><br/>Main Street 1<br/>AA 25012 Exampleville <br/><b>Phone:</b> 555-555-555', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a PDF from a json or csv file, or a json or csv string"
    orientation: Orientation of pdf (portrait/landscape)
        url: The (public) url to get the csv file from
        filename: The filename of the pdf to be created
        titlealign: Horizontal Alignment of the title
        subtitlealign: Horizontal alignment of the subtitle
        highlightfirstcolumn: Highlight the first column
        highlightlastcolumn: Highligh the last column
        data: Instead of a url, you can pass the json or csv string to this parameter directly
        template: The template for the layout of the pdf.  Available templates are : relax,candy,green,red and pink
        separator: A character that is the separator for the csv data.  By default ';' is used.
        subtitle: The subtitle. Following html tags are allowed : <b></b>,<i></i>,<ol><li></li></ol>, <ul><li></li></ul>,<p></p>,<br/>,<br>.
        title: The title. Following html tags are allowed : <b></b>,<i></i>,<ol><li></li></ol>, <ul><li></li></ul>,<p></p>,<br/>,<br>.
        
    """
    url = f"https://easypdf.p.rapidapi.com/pdfConvert"
    querystring = {'orientation': orientation, 'url': url, }
    if filename:
        querystring['filename'] = filename
    if titlealign:
        querystring['titleAlign'] = titlealign
    if subtitlealign:
        querystring['subtitleAlign'] = subtitlealign
    if highlightfirstcolumn:
        querystring['highlightFirstColumn'] = highlightfirstcolumn
    if highlightlastcolumn:
        querystring['highlightLastColumn'] = highlightlastcolumn
    if data:
        querystring['data'] = data
    if template:
        querystring['template'] = template
    if separator:
        querystring['separator'] = separator
    if subtitle:
        querystring['subtitle'] = subtitle
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easypdf.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

