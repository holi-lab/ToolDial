INPUT_DESC_PROMPT = """
You are an intelligent annotator.
Your mission is to write the description of input parameters more specific, refering the give information.

Write as specific as possible refering the given information.
The new description should be based on the existing description but written in a way that better reflects the content of the API description and API endpoint description than before.
Just return the input and it's description. Not a single words

For example:

Category of the API: Data
Description about the Category: APIs facilitate the seamless exchange of data between applications and databases, enabling developers to integrate functionalities securely and swiftly.
API Name: YouTube Media Downloader
API description: A scraper API for YouTube search and download. Get videos, subtitles, comments without age or region limits (proxy URL supported).
API endpoint name: Get Channel Details
API endpoint description: This endpoint fetches details of a YouTube channel.

List of input parameters:
input parameter name: channelId
description: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.

input parameter name: lang
description: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.

For this, you should return
[["channelId","The unique identifier for the YouTube channel, which can be the channel's ID, a custom URL name, or a channel handle. When using a channel handle, ensure to prefix it with `@` (e.g., `@channelname`)."],["lang","The language code (ISO-639) used to specify the language for the localized results. If not provided, the default is `en-US`. In case an unsupported language code is supplied, the results will revert to `en-US`."]]

Now I'll give you another description. Follow the instruction, refering the example.

Write as specific as possible refering the given information.
The new description should be based on the existing description but written in a way that better reflects the content of the API description and API endpoint description than before.
Just return the input and it's description. Not a single words
"""

OUTPUT_DESC_PROMPT = """
You are an intelligent annotator. Your mission is to write the description of the output components of some API endpoint, refering the given information below.

For example,

Category of the API: Data
Description about the Category: APIs facilitate the seamless exchange of data between applications and databases, enabling developers to integrate functionalities securely and swiftly.
API Name: YouTube Media Downloader
API description: A scraper API for YouTube search and download. Get videos, subtitles, comments without age or region limits (proxy URL supported).
API endpoint name: Get Channel Details
API endpoint description: This endpoint fetches details of a YouTube channel.

Based on the given description, write the description about the output component of this API endpoint. Write as specific as possible. Don't generate the example of each components.
The description should reflect as much as possible the description of the API and the API endpoint, so that even someone seeing this API endpoint for the first time can understand exactly what the output component means.
(Component which is sepearted with | refers the hierarchy of the schema. For example, avatar|height means the height of the avater.)

Output components:
[
    {
        "name": "status"
    },
    {
        "name": "type"
    },
    {
        "name": "id"
    },
    {
        "name": "name"
    },
    {
        "name": "handle"
    },
    {
        "name": "description"
    },
    {
        "name": "isVerified"
    },
    {
        "name": "isVerifiedArtist"
    },
    {
        "name": "subscriberCountText"
    },
    {
        "name": "videoCountText"
    },
    {
        "name": "viewCountText"
    },
    {
        "name": "joinedDateText"
    },
    {
        "name": "country"
    },
    {
        "name": "links|title"
    },
    {
        "name": "links|url"
    },
    {
        "name": "avatar|url"
    },
    {
        "name": "avatar|width"
    },
    {
        "name": "avatar|height"
    }
]


For this example, you have to return,

```json
[
    {
        "name": "status",
        "description": "Indicates whether the API call was successful. True means the call was successful, while False means it failed"
    },
    {
        "name": "type",
        "description": "Specifies the type of YouTube channel, such as 'User' or 'Brand', indicating the category of the channel."
    },
    {
        "name": "id",
        "description": "The unique identifier assigned to the YouTube channel, which can be used to reference the channel in other API calls or services."
    },
    {
        "name": "name",
        "description": "The official name of the YouTube channel as displayed on the platform, which is set by the channel owner."
    },
    {
        "name": "handle",
        "description": "The unique handle of the YouTube channel, which often appears in the URL of the channel's page."
    },
    {
        "name": "description",
        "description": "A textual description provided by the channel owner that gives an overview of the channel’s content, themes, and purpose."
    },
    {
        "name": "isVerified",
        "description": "Indicates whether the YouTube channel is verified by YouTube. A verified status signifies authenticity and is usually granted to public figures, brands, and popular content creators."
    },
    {
        "name": "isVerifiedArtist",
        "description": "Specifies if the YouTube channel is recognized as a verified artist's channel, which is a special status for musicians and bands to highlight their official content."
    },
    {
        "name": "subscriberCountText",
        "description": "A human-readable representation of the number of subscribers the channel has, formatted for display purposes."
    },
    {
        "name": "videoCountText",
        "description": "A human-readable representation of the total number of videos uploaded by the channel, formatted for display purposes."
    },
    {
        "name": "viewCountText",
        "description": "A human-readable representation of the total number of views across all videos on the channel, formatted for display purposes."
    },
    {
        "name": "joinedDateText",
        "description": "A human-readable representation of the date when the YouTube channel was created, formatted for display purposes."
    },
    {
        "name": "country",
        "description": "The country where the YouTube channel is registered or primarily based, providing geographical context."
    },
    {
        "name": "links|title",
        "description": "The title of an external link provided by the channel, which can lead to the channel’s social media profiles, websites, or other related content."
    },
    {
        "name": "links|url",
        "description": "The URL of an external link associated with the channel, which directs users to other online presences of the channel."
    },
    {
        "name": "avatar|url",
        "description": "The URL of the channel's avatar image, which is the profile picture displayed on the channel's page."
    },
    {
        "name": "avatar|width",
        "description": "The width of the avatar image in pixels, providing information about the image dimensions."
    },
    {
        "name": "avatar|height",
        "description": "The height of the avatar image in pixels, providing information about the image dimensions."
    }
]
```

 
Now, I'll give you another API endpoint's description. Write the description about the output components and return it as a same format of the example. Just return the result. Not a single words.
Based on the given description, write the description about the output component of this API endpoint. Write as specific as possible. Don't generate the example of each components.
The description should reflect as much as possible the description of the API and the API endpoint, so that even someone seeing this API endpoint for the first time can understand exactly what the output component means.
(Component which is sepearted with | refers the hierarchy of the schema. For example, avatar|height means the height of the avater.)

Fill the <Your response>.

```json
<Your response>
```

"""
