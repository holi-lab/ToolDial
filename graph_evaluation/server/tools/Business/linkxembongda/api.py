import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def link_xem_b_ng(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Link xem bóng đá tv trực tuyến mới nhất hôm nay. Link xem bóng đá không bị chặn, chia sẻ tổng review các đường link xem tốt nhất.
		#linkxembongda #linkbongdalau #linktructiepbongda #linkvaoxembong #linkxembongda.live #reviewwebbongda #danhgiatrangwebbongdab#webbongdanhieu 
		Địa chỉ 200 Trần Đăng Ninh, Dịch Vọng, Cầu Giấy, Hà Nội 
		Hotline 0866767785
		Gmail: linkxembongdalive@gmail.com
		Website https://linkxembongda.live/ 
		Google map https://www.google.com/maps?cid=7363081607037732360 
		https://linkxembongda.live/thu-muc/web-bong-xem-nhieu/ 
		https://linkxembongda.live/thu-muc/web-bong-da-moi/ 
		https://linkxembongda.live/thu-muc/tin-tuc/ 
		https://www.facebook.com/linkxembongdalive/ 
		https://communities.bentley.com/members/1195bf0b_2d00_6f05_2d00_44c2_2d00_b5a6_2d00_09e1d5cb8c11
		https://pxhere.com/en/photographer/3843144
		https://www.cplusplus.com/user/linkxembongda
		https://www.picfair.com/users/linkxembongda
		https://replit.com/@linkxembongda
		https://leetcode.com/linkxembongda/
		https://www.intensedebate.com/people/linkxembongda1
		https://postheaven.net/linkxembongda/link-xem-bong-da
		https://www.laonsw.net/web/linkxembongda/home/-/blogs/link-xem-bong-a
		https://platform.xr4all.eu/-/snippets/1681
		https://justpaste.it/u/linkxembongda
		https://pantip.com/profile/7064081#topics
		https://www.emoneyspace.com/linkxembongda
		https://www.veoh.com/users/linkxembongdalive
		https://app.glosbe.com/profile/6938127546907102425
		https://hdvietnam.org/members/linkxembongda.2074018/
		https://seedandspark.com/user/linkxembongda-1
		https://meadd.com/bongdalive/72183130
		https://appsliced.co/u/linkxembongda
		https://calis.delfi.lv/blogs/posts/81999-link-xem-bong-da/lietotajs/262485-linkxembongda/"
    
    """
    url = f"https://linkxembongda.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkxembongda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

