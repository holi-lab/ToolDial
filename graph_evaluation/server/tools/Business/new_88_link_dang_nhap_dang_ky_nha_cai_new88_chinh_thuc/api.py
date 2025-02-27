import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_88_link_dang_nhap_dang_ky_nha_cai_new88_chinh_thuc(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New88 - new88vinet là nhà cái cá cược trực tuyến uy tín top đầu ở khu vực Việt Nam và Đông Nam Á.
		Các sản phẩm cá cược được kiểm duyệt và phân phối độc quyền bởi tập đoàn Taipei Đài Loan. 
		#New88 #linkvao #dangnhap #dangky #lnhacaiNew882022 #khuyenmainew88 #new88vinet
		Hotline 0988.366.638
		Địa chỉ : 134 Đ. Bạch Đằng, Chương Dương Độ, Hoàn Kiếm, Hà Nội, Việt Nam
		GG Map https://www.google.com/maps?cid=4292596333329558795 
		Gmail: new88vi@gmail.com
		Website https://new88vi.net/ 
		https://new88vi.net/dang-ky-new88/ 
		https://new88vi.net/khuyen-mai-new88/ 
		https://www.facebook.com/new88vi 
		https://sites.google.com/view/new88vi 
		https://www.myminifactory.com/users/new88vi
		https://www.surfaceforums.net/members/new88vi.42875/
		https://forum.acronis.com/user/410259
		https://www.instapaper.com/p/10942820
		https://gitlab.tails.boum.org/new88vi
		https://www.dualmonitorbackgrounds.com/new88vi
		https://muckrack.com/new88-vi
		https://roundme.com/@new88vi/about
		https://ioby.org/users/new88vi652831
		https://www.joindota.com/users/2047642-new88vi
		http://prsync.com/newvinet/
		https://community.aodyo.com/user/new88vi
		https://mmo4me.com/members/new88vi.197888/
		https://myanimelist.net/profile/new88vi
		https://communities.bentley.com/members/4dfd36dd_2d00_31a2_2d00_4670_2d00_829b_2d00_71bc83ca1d07
		https://pxhere.com/en/photographer/3860382
		https://www.cplusplus.com/user/new88vi
		https://www.picfair.com/users/new88vi
		https://replit.com/@new88vi
		https://leetcode.com/new88vi/
		https://www.intensedebate.com/profiles/new88vi
		http://cloudsdeal.xobor.de/u7801_new-vi.html
		https://platform.xr4all.eu/new88vi
		https://justpaste.it/u/new88vi
		https://pantip.com/profile/7131279#topics
		https://www.emoneyspace.com/new88vi
		https://www.veoh.com/users/new88vi
		https://app.glosbe.com/profile/6954760304760720592
		https://www.bandlab.com/user7937408780351553
		https://seedandspark.com/user/new88vi
		https://meadd.com/new88vi
		https://appsliced.co/u/new88vi
		https://vocal.media/authors/new88vi
		https://8tracks.com/new88vi
		https://coub.com/new88vi14468
		https://app.roll20.net/users/10776874/new88-v"
    
    """
    url = f"https://new-88-link-dang-nhap-dang-ky-nha-cai-new88-chinh-thuc.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-88-link-dang-nhap-dang-ky-nha-cai-new88-chinh-thuc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

