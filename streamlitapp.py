import streamlit as st
# from audio_recorder_streamlit import audio_recorder
from streamlit_modal import Modal
import requests

url='http://13.234.19.201:8505/'


st.set_page_config(layout="wide",page_title="LoopTech HR",page_icon="🤖" )



settings=st.popover("settings")
with settings:
    querying_tech = settings.radio(
    "querying technic",
    ["normal","multiquery", "ragfusion", "decomposition","stepback","hyde"])

    retrieving_tech = settings.radio(
    "retrieving technic",
    ["colbert","chroma", "raptor"])




if 'store_message' not in st.session_state:
    st.session_state.store_message=[]

st.title(":blue[LOOPTECH] HR POLICY HELPER")

container=st.container(height=400)
message=st.chat_input("say somthing.....")

# col1, col2 = st.columns([1,9])

# with col1:
#     audio_bytes=audio_recorder(text="",
#     recording_color="#e8b62c",
#     neutral_color="#6aa36f",
#     icon_size="2x")

# with col2:
#     message=st.chat_input("say somthing.....")



demo_chat=container.chat_message(name="assistant",avatar="🤖")
with demo_chat:
    demo_chat.write("Hi There!, I am your AI BOT, How can I help ?")



# if audio_bytes:
#     print("hai")
#     audio_location="temp.mp3"
#     with open(audio_location,"wb") as f:
#         f.write(audio_bytes)
#         print("successfull")

    
if(message):
    
    data={

        "query":message,
        "queryingtech":querying_tech,
        "retrievingtech":retrieving_tech
    } 

    response= requests.post(url=url,json=data)
    
    if(response.status_code==200):
        response_data=response.json()
        st.session_state.store_message.append({"user":message,"ass":response_data['response']}) 
    else:
        model=Modal(key="Demo Key",title="server Error")
        with model.container():
            st.write("SOMETHING WRONG WITH A SERVER")
            st.write(response.status_code)

    



for chat in st.session_state.store_message:
    user=container.chat_message(name="user",avatar="👨🏻‍💼")
    with user:
        user.write(chat['user'])
    ass=container.chat_message(name="assistant",avatar="🤖")
    with ass:
        ass.write(chat['ass'])