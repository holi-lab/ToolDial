import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert(document_url: str, document_name: str='pdflayer.pdf', custom_unit: str='px', user_agent: str=None, accept_lang: str='en-US', text_encoding: str='utf-8', auth_user: str=None, auth_pass: str=None, owner_password: str=None, user_password: str=None, page_size: str='A4', orientation: str='portrait', footer_spacing: str=None, footer_text: str=None, footer_url: str=None, viewport: str='1440x900', css_url: str=None, delay: str=None, zoom: int=None, watermark_url: str=None, watermark_offset_x: int=None, watermark_offset_y: int=None, watermark_opacity: int=20, watermark_in_background: int=None, title: str=None, subject: str=None, creator: str='pdflayer.com', author: str=None, no_javascript: int=None, header_align: str='center', header_url: str=None, footer_align: str='center', ttl: int=2592000, force: int=None, inline: int=None, encryption: int=None, no_images: int=None, no_hyperlinks: int=None, no_backgrounds: int=None, use_print_media: int=None, grayscale: int=None, low_quality: int=None, forms: int=None, no_print: int=None, no_modify: int=None, no_copy: int=None, page_width: int=None, page_height: int=None, margin_top: int=None, margin_bottom: int=None, margin_left: int=None, margin_right: int=None, header_spacing: int=None, header_text: int=None, dpi: int=96, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Main API endpoint for PDF conversion"
    document_url: The URL of the HTML source you would like to convert to PDF. Looking to use pass raw HTML code to the API? Learn more at https://pdflayer.com/documentation#quickstart
        document_name: specify a PDF name of up to 180 characters.
        custom_unit: set to px (Pixels), pt (Points), in (Inches) or mm (Millimeters)
        user_agent: set to your preferred User-Agent header string
        accept_lang: set to your preferred Accept-Language header string
        text_encoding: set to your preferred text encoding string
        auth_user: specify username used to access password-protected site
        auth_pass: specify password used to access password-protected site
        owner_password: specify owner password to password protect PDF
        user_password: specify user password to password protect PDF
        page_size: set to preferred page size, e.g. A4, A5, etc.
        orientation: set to portrait or landscape
        footer_spacing: set to preferred footer spacing value (numeric), e.g. 10
        footer_text: set to preferred footer text, e.g. "This is my footer"
        footer_url: set to (urlencoded) URL containing your preferred footer HTML elements. Want to use raw HTML code? Learn more at https://pdflayer.com/documentation#pdf_footer
        viewport: Set to preferred viewport "width x height", e.g. 320x480
        css_url: inject a custom CSS stylesheet using a (urlencoded) URL
        delay: specify a delay (in seconds) before PDF is captured
        zoom: specify page zoom factor between 0 and 50
        watermark_url: specify a watermark URL (urlencoded) containing a PNG or JPG image
        watermark_offset_x: specify a horizontal watermark offset, e.g. 10
        watermark_offset_y: specify a vertical watermark offset, e.g. 10
        watermark_opacity: specify watermark opacity percentage (numeric) between 0 and 100
        watermark_in_background: set to 1 to place watermark behind text
        title: specify a PDF document title of max. 150 characters
        subject: specify a PDF document subject of max. 150 characters
        creator: specify a PDF document creator name of max. 150 characters
        author: specify a PDF document author name of max. 150 characters
        no_javascript: Set to 1 in order to disable JavaScript
        header_align: set to left, center or right
        header_url: set to (urlencoded) URL containing your preferred header HTML elements. Want to use raw HTML code? Learn more at https://pdflayer.com/documentation#pdf_header
        footer_align: set to left, center or right
        ttl: the time (in seconds) a generated PDF is cached
        force: set to 1 to force new PDF
        inline: set to 1 to display PDF document inline
        encryption: set to 40 (40-bit) or 128 (128-bit)
        no_images: Set to 1 in order to disable images
        no_hyperlinks: Set to 1 in order to disable hyperlinks
        no_backgrounds: Set to 1 in order to disable CSS backgrounds
        use_print_media: Set to 1 in order to activate CSS @media print declarations
        grayscale: Set to 1 in order to remove all colours
        low_quality: Set to 1 in order to generate low quality PDF
        forms: Set to 1 in order to enable forms on your PDF
        no_print: Set to 1 in order to disable printing of the final PDF document. Requires encryption, owner_password or user_password to be specified
        no_modify: Set to 1 in order to disable modification of the final PDF document. Requires encryption, owner_password or user_password to be specified
        no_copy: Set to 1 in order to disable the possibility to copy any text of the final PDF document. Requires encryption, owner_password or user_password to be specified
        page_width: specify page width (numeric), e.g. 200 (overrides page_size)
        page_height: specify page height (numeric), e.g. 600 (overrides page_size)
        margin_top: set to preferred top margin value (numeric), e.g. 5
        margin_bottom: set to preferred bottom margin value (numeric), e.g. 5
        margin_left: set to preferred left margin value (numeric), e.g. 5
        margin_right: set to preferred right margin value (numeric), e.g. 5
        header_spacing: set to preferred header spacing value (numeric), e.g. 10
        header_text: set to preferred header text, e.g. "This is my heading"
        dpi: specify the DPI resolution (numerical) between 10 and 10000
        
    """
    url = f"https://apilayer-pdflayer-v1.p.rapidapi.com/convert"
    querystring = {'document_url': document_url, }
    if document_name:
        querystring['document_name'] = document_name
    if custom_unit:
        querystring['custom_unit'] = custom_unit
    if user_agent:
        querystring['user_agent'] = user_agent
    if accept_lang:
        querystring['accept_lang'] = accept_lang
    if text_encoding:
        querystring['text_encoding'] = text_encoding
    if auth_user:
        querystring['auth_user'] = auth_user
    if auth_pass:
        querystring['auth_pass'] = auth_pass
    if owner_password:
        querystring['owner_password'] = owner_password
    if user_password:
        querystring['user_password'] = user_password
    if page_size:
        querystring['page_size'] = page_size
    if orientation:
        querystring['orientation'] = orientation
    if footer_spacing:
        querystring['footer_spacing'] = footer_spacing
    if footer_text:
        querystring['footer_text'] = footer_text
    if footer_url:
        querystring['footer_url'] = footer_url
    if viewport:
        querystring['viewport'] = viewport
    if css_url:
        querystring['css_url'] = css_url
    if delay:
        querystring['delay'] = delay
    if zoom:
        querystring['zoom'] = zoom
    if watermark_url:
        querystring['watermark_url'] = watermark_url
    if watermark_offset_x:
        querystring['watermark_offset_x'] = watermark_offset_x
    if watermark_offset_y:
        querystring['watermark_offset_y'] = watermark_offset_y
    if watermark_opacity:
        querystring['watermark_opacity'] = watermark_opacity
    if watermark_in_background:
        querystring['watermark_in_background'] = watermark_in_background
    if title:
        querystring['title'] = title
    if subject:
        querystring['subject'] = subject
    if creator:
        querystring['creator'] = creator
    if author:
        querystring['author'] = author
    if no_javascript:
        querystring['no_javascript'] = no_javascript
    if header_align:
        querystring['header_align'] = header_align
    if header_url:
        querystring['header_url'] = header_url
    if footer_align:
        querystring['footer_align'] = footer_align
    if ttl:
        querystring['ttl'] = ttl
    if force:
        querystring['force'] = force
    if inline:
        querystring['inline'] = inline
    if encryption:
        querystring['encryption'] = encryption
    if no_images:
        querystring['no_images'] = no_images
    if no_hyperlinks:
        querystring['no_hyperlinks'] = no_hyperlinks
    if no_backgrounds:
        querystring['no_backgrounds'] = no_backgrounds
    if use_print_media:
        querystring['use_print_media'] = use_print_media
    if grayscale:
        querystring['grayscale'] = grayscale
    if low_quality:
        querystring['low_quality'] = low_quality
    if forms:
        querystring['forms'] = forms
    if no_print:
        querystring['no_print'] = no_print
    if no_modify:
        querystring['no_modify'] = no_modify
    if no_copy:
        querystring['no_copy'] = no_copy
    if page_width:
        querystring['page_width'] = page_width
    if page_height:
        querystring['page_height'] = page_height
    if margin_top:
        querystring['margin_top'] = margin_top
    if margin_bottom:
        querystring['margin_bottom'] = margin_bottom
    if margin_left:
        querystring['margin_left'] = margin_left
    if margin_right:
        querystring['margin_right'] = margin_right
    if header_spacing:
        querystring['header_spacing'] = header_spacing
    if header_text:
        querystring['header_text'] = header_text
    if dpi:
        querystring['dpi'] = dpi
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-pdflayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

