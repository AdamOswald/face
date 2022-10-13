from nextcord.client.client import Client
from nextcord.core.http import Route
from nextcord.flags import Intents
from os import environ as env
from aiohttp import ClientSession
from asyncio import sleep
from json import dumps
from random import randint

intents = Intents()
intents.value = 512

color_scheme = [
    0xF34213,
    0x1098F7,
    0xFFBC42,
    0x8F2D56,
    0x31393C,
    0xCA1551,
    0xEAC435,
    0x345995,
    0x03CEA4,
    0xFB4D3D,
    0x2AB7CA,
    0x399E5A,
    0x98CCEE
]
themes = {
    1: "Synthwave",
    2: "Paperius",
    3: "Experimental Plant",
    4: "Steampunk",
    5: "Dark Reality",
    6: "Candy",
    7: "Wooden",
    8: "Desert",
    9: "Psychic",
    12: "Christmas"
}
current_active = 0

client = Client(env["TOKEN"], intents)
@client.state.gateway.event_dispatcher.listen("MESSAGE_CREATE")
async def on_message(_, message: dict):
    global current_active
    content: str = message["content"]
    if content is None:
        return
    if content.startswith("*art "):
        prompt = content.removeprefix("*art ")
        
        splitted = prompt.split(" ")
        try:
            style = int(splitted[0])
        except:
            style = randint(0,12)
        finally:
            if style > 12 or style < 0:
                style = randint(0, 12)
            else:
                prompt = " ".join(splitted[1:])
        current_active += 1
        try:
            image_url = await get_image(prompt, style)
        except:
            current_active -= 1
            raise
        current_active -= 1
        route = Route("POST", "/channels/{channel_id}/messages", channel_id=message["channel_id"])
        message_data = {
            "message_reference": {
                "message_id": message["id"],
                "guild_id": message["guild_id"],
                "fail_if_not_exists": False
            },
            "embeds": [{
                "image": {"url": image_url},
                "color": color_scheme[style],
                "title": themes.get(style, f"#{style}")
            }],
        }
        # await client.state.http.request(route, json={"content": image_url})
        await client.state.http.request(route, json=message_data)
    elif content.startswith("*styles"):
        style_info = []
        for style_id in range(1,13):
            style_info.append(f"**{style_id}.** {themes.get(style_id, '?')}")
        route = Route("POST", "/channels/{channel_id}/messages", channel_id=message["channel_id"])
        await client.state.http.request(route, json={"content": "\n".join(style_info)})
    elif content.startswith("*status"):
        route = Route("POST", "/channels/{channel_id}/messages", channel_id=message["channel_id"])
        await client.state.http.request(route, json={"content": f"Current active: {current_active}"}) 
       

   
async def get_image(prompt: str, style: int):
    # Get images
    session: ClientSession = client.state.http._session

    ## Auth
    r = await session.post("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyDCvp5MTJLUdtBYEKYWXJrlLzu1zuKM6Xw", json={"returnSecureToken": True})
    data = await r.json()
    token = data["idToken"]
    auth_headers = {"Authorization": "bearer " + token}

    # Get a task id
    r = await session.post("https://app.wombo.art/api/tasks", headers=auth_headers, json=dumps({"premium": False}))
    data = await r.json()

    task_id = data["id"]
    
    # Start the task
    query = {"input_spec": {
        "display_freq": 10,  # Not sure what this does exactly
        "prompt": prompt,
        "style": style  # Not sure how many there are, theres atleast 6
    }}
    r = await session.put("https://app.wombo.art/api/tasks/" + task_id, json=dumps(query), headers=auth_headers)
    data = await r.json()

    # Wait until completed
    while True:
        r = await session.get("https://app.wombo.art/api/tasks/" + task_id, headers=auth_headers)
        data = await r.json()
        state = data["state"]

        if state == "completed":
            break 
        if state == "failed":
            print(data)
            raise RuntimeError(data)

        await sleep(1)
    
    
    image_url = data["photo_url_list"][-1]
    return image_url
     

client.run()
