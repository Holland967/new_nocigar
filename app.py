from openai import OpenAI
import streamlit as st
import os

from model import *
from template import *

password: str = os.getenv("PASSWORD")

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    password_input: str = st.text_input(
        label="Password",
        value="",
        type="password",
        key="password_input")
    login_button: bool = st.button(
        label="Login",
        key="login_button",
        type="primary")
    if login_button:
        if password_input == password:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Incorrect password")

if st.session_state.login:
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY")
    siliconflow_base_url: str = os.getenv("SILICONFLOW_BASE_URL")
    yi_api_key: str = os.getenv("YI_API_KEY")
    yi_base_url: str = os.getenv("YI_BASE_URL")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL")

    with st.sidebar:
        func_select: str = st.selectbox(
            label="Function",
            options=[
                "General Chat",
                "Image Chat",
                "Spider Chat",
                "Translator"],
            index=0,
            key="func_select")
        
        clear_button: bool = st.button(
            label="Clear",
            key="clear_button",
            type="primary",
            use_container_width=True)
        undo_button: bool = st.button(
            label="Undo",
            key="undo_button",
            use_container_width=True)
        retry_button: bool = st.button(
            label="Retry",
            key="retry_button",
            use_container_width=True)

    if func_select == "General Chat":
        st.subheader("General Chat", anchor=False)

        if "general_sys" not in st.session_state:
            st.session_state.general_sys = default_prompt
        if "general_chat_msg" not in st.session_state:
            st.session_state.general_chat_msg = []
        if "general_chat_history" not in st.session_state:
            st.session_state.general_chat_history = []
        if "general_retry_state" not in st.session_state:
            st.session_state.general_retry_state = False
        
        if "general_lock" not in st.session_state:
            st.session_state.general_lock = True
        
        if st.session_state.general_chat_msg != []:
            st.session_state.general_lock = False
        else:
            st.session_state.general_lock = True
        
        with st.sidebar:
            server_select: str = st.selectbox(
                label="Server Provider",
                options=[
                    "Siliconflow",
                    "Lingyiwanwu",
                    "DeepSeek"],
                index=0,
                key="server_select",
                disabled=not st.session_state.general_lock)

            if server_select == "Siliconflow":
                model_list = siliconflow_model
                temperature_value = 0.7
                topp_value = 0.7
                api_key = siliconflow_api_key
                base_url = siliconflow_base_url
            elif server_select == "Lingyiwanwu":
                model_list = yi_model
                temperature_value = 0.3
                topp_value = 0.9
                api_key = yi_api_key
                base_url = yi_base_url
            elif server_select == "DeepSeek":
                model_list = deepseek_model
                temperature_value = 1.0
                topp_value = 1.0
                api_key = deepseek_api_key
                base_url = deepseek_base_url
            
            model: str = st.selectbox(
                label="Model",
                options=model_list,
                index=0,
                key="general_model_select",
                disabled=not st.session_state.general_lock)
            
            system_prompt: str = st.text_area(
                label="System Prompt",
                value=st.session_state.general_sys,
                key="general_system_prompt",
                disabled=not st.session_state.general_lock)
            
            max_tokens: int = st.slider(
                label="Max Tokens",
                min_value=1,
                max_value=4096,
                value=4096,
                key="general_max_tokens",
                disabled=not st.session_state.general_lock)
            
            temperature: float = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=2.0,
                value=temperature_value,
                step=0.01,
                key="general_temperature",
                disabled=not st.session_state.general_lock)
            
            top_p: float = st.slider(
                label="Top P",
                min_value=0.0,
                max_value=1.0,
                value=topp_value,
                step=0.01,
                key="general_top_p",
                disabled=not st.session_state.general_lock)
            
            frequency_penalty: float = st.slider(
                label="Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="general_frequency_penalty",
                disabled=not st.session_state.general_lock)
            
            presence_penalty: float = st.slider(
                label="Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="general_presence_penalty",
                disabled=not st.session_state.general_lock)
        
        for i in st.session_state.general_chat_history:
            with st.chat_message(i["role"]):
                st.markdown(i["content"])
        
        if query := st.chat_input("Say something...", key="query"):
            st.session_state.general_chat_msg.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            messages: list = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.general_chat_msg
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.general_chat_msg.append({"role": "assistant", "content": result})
            st.session_state.general_chat_history = st.session_state.general_chat_msg
            st.rerun()
        
        if clear_button:
            st.session_state.general_sys = default_prompt
            st.session_state.general_chat_msg = []
            st.session_state.general_chat_history = []
            st.rerun()
        
        if undo_button:
            del st.session_state.general_chat_msg[-1]
            del st.session_state.general_chat_history[-1]
            st.rerun()
        
        if retry_button:
            st.session_state.general_chat_msg.pop()
            st.session_state.general_chat_history = []
            st.session_state.general_retry_state = True
            st.rerun()
        if st.session_state.general_retry_state:
            for i in st.session_state.general_chat_msg:
                with st.chat_message(i["role"]):
                    st.markdown(i["content"])
            messages: list = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.general_chat_msg
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.general_chat_msg.append({"role": "assistant", "content": result})
            st.session_state.general_chat_history = st.session_state.general_chat_msg
            st.session_state.general_retry_state = False
            st.rerun()
        
    elif func_select == "Image Chat":
        from image import process_img
        import requests
        import base64

        api_key: str = siliconflow_api_key
        base_url: str = siliconflow_base_url

        st.subheader("Image Chat", anchor=False)

        if "source_state" not in st.session_state:
            st.session_state.source_state = False

        if "image_chat_msg" not in st.session_state:
            st.session_state.image_chat_msg = []
        if "image_chat_history" not in st.session_state:
            st.session_state.image_chat_history = []
        if "image_retry_state" not in st.session_state:
            st.session_state.image_retry_state = False
        
        if "image_lock" not in st.session_state:
            st.session_state.image_lock = True
        
        if st.session_state.image_chat_msg != []:
            st.session_state.image_lock = False
        else:
            st.session_state.image_lock = True
        
        with st.sidebar:
            model_list: list = vision_model
            model: str = st.selectbox(
                label="Model",
                options=model_list,
                index=0,
                key="image_model_select",
                disabled=not st.session_state.image_lock)

            system_prompt: str = st.text_area(
                label="System Prompt",
                value="",
                key="image_system_prompt",
                disabled=not st.session_state.image_lock)
            
            max_tokens: int = st.slider(
                label="Max Tokens",
                min_value=1,
                max_value=4096,
                value=4096,
                key="image_max_tokens",
                disabled=not st.session_state.image_lock)
            
            temperature: float = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.01,
                key="image_temperature",
                disabled=not st.session_state.image_lock)
            
            top_p: float = st.slider(
                label="Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01,
                key="image_top_p",
                disabled=not st.session_state.image_lock)
            
            frequency_penalty: float = st.slider(
                label="Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="image_frequency_penalty",
                disabled=not st.session_state.image_lock)
            
            presence_penalty: float = st.slider(
                label="Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="image_presence_penalty",
                disabled=not st.session_state.image_lock)
        
        img_file = st.file_uploader(
            label="Upload an image",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
            accept_multiple_files=False,
            disabled=not st.session_state.image_lock,
            label_visibility="collapsed")
        if img_file is not None:
            img_data = img_file.read()
            processed_data = process_img(img_data)
            processed_data = base64.b64encode(processed_data).decode("utf-8")
            processed_data = f"data:image/png;base64,{processed_data}"
        
        img_url: str = st.text_input(
            label="Image URL",
            value="",
            key="image_url",
            disabled=not st.session_state.image_lock)
        if img_url != "":
            img_data = requests.get(img_url).content
            processed_data = process_img(img_data)
            processed_data = base64.b64encode(processed_data).decode("utf-8")
            processed_data = f"data:image/png;base64,{processed_data}"
        
        if img_file is not None and img_url == "":
            if processed_data:
                st.session_state.source_state = True
        elif img_file is None and img_url != "":
            if processed_data:
                st.session_state.source_state = True
        elif img_file is None and img_url == "":
            st.session_state.source_state = False
        elif img_file is not None and img_url!= "":
            st.error("Please only upload an image or input an image URL!")
            st.session_state.source_state = False
        
        if st.session_state.source_state:
            with st.expander("Image Preview"):
                st.image(processed_data)
        
        for i in st.session_state.image_chat_history:
            with st.chat_message(i["role"]):
                st.markdown(i["content"])
        
        if query := st.chat_input(
            placeholder="Say something...",
            key="image_query",
            disabled=not st.session_state.source_state):
            st.session_state.image_chat_msg.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            messages: list = []
            if system_prompt != "":
                sys_msg = {"role": "system", "content": system_prompt}
                messages.append(sys_msg)
            if len(st.session_state.image_chat_msg) == 1:
                user_msg = {
                    "role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": processed_data}}]}
                messages.append(user_msg)
                messages: list = messages
            else:
                user_msg = {
                    "role": "user", "content": [
                        {"type": "text", "text": st.session_state.image_chat_msg[0]["content"]},
                        {"type": "image_url", "image_url": {"url": processed_data}}]}
                messages.append(user_msg)
                messages: list = messages + st.session_state.image_chat_msg[1:]
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.image_chat_msg.append({"role": "assistant", "content": result})
            st.session_state.image_chat_history = st.session_state.image_chat_msg
            st.rerun()
        
        if clear_button:
            st.session_state.image_chat_msg = []
            st.session_state.image_chat_history = []
            st.rerun()
        
        if undo_button:
            del st.session_state.image_chat_msg[-1]
            del st.session_state.image_chat_history[-1]
            st.rerun()
        
        if retry_button:
            st.session_state.image_chat_msg.pop()
            st.session_state.image_chat_history = []
            st.session_state.image_retry_state = True
            st.rerun()
        if st.session_state.image_retry_state:
            for i in st.session_state.image_chat_msg:
                with st.chat_message(i["role"]):
                    st.markdown(i["content"])
            messages: list = []
            if system_prompt != "":
                sys_msg = {"role": "system", "content": system_prompt}
                messages.append(sys_msg)
            if len(st.session_state.image_chat_msg) == 1:
                user_msg = {
                    "role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": processed_data}}]}
                messages.append(user_msg)
                messages: list = messages
            else:
                user_msg = {
                    "role": "user", "content": [
                        {"type": "text", "text": st.session_state.image_chat_msg[0]["content"]},
                        {"type": "image_url", "image_url": {"url": processed_data}}]}
                messages.append(user_msg)
                messages: list = messages + st.session_state.image_chat_msg[1:]
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.image_chat_msg.append({"role": "assistant", "content": result})
            st.session_state.image_chat_history = st.session_state.image_chat_msg
            st.session_state.image_retry_state = False
            st.rerun()
    
    elif func_select == "Spider Chat":
        from spider import Spider

        spider = Spider()

        api_key: str = siliconflow_api_key
        base_url: str = siliconflow_base_url

        st.subheader("Spider Chat", anchor=False)

        if "spider_sys" not in st.session_state:
            st.session_state.spider_sys = spider_prompt

        if "spider_chat_msg" not in st.session_state:
            st.session_state.spider_chat_msg = []
        if "spider_chat_history" not in st.session_state:
            st.session_state.spider_chat_history = []
        if "spider_retry_state" not in st.session_state:
            st.session_state.spider_retry_state = False
        
        if "spider_lock" not in st.session_state:
            st.session_state.spider_lock = True

        if "web_content" not in st.session_state:
            st.session_state.web_content = ""
        if "continue_" not in st.session_state:
            st.session_state.continue_ = False
        
        if st.session_state.spider_chat_msg != []:
            st.session_state.spider_lock = False
        else:
            st.session_state.spider_lock = True
        
        with st.sidebar:
            model_list: list = spider_model
            model: str = st.selectbox(
                label="Model",
                options=model_list,
                index=0,
                key="spider_model_select",
                disabled=not st.session_state.spider_lock)
            
            system_prompt: str = st.text_area(
                label="System Prompt",
                value=st.session_state.spider_sys,
                key="spider_system_prompt",
                disabled=not st.session_state.spider_lock)
            st.session_state.spider_sys = system_prompt
            
            max_tokens: int = st.slider(
                label="Max Tokens",
                min_value=1,
                max_value=4096,
                value=4096,
                key="spider_max_tokens",
                disabled=not st.session_state.spider_lock)
            
            temperature: float = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.01,
                key="spider_temperature",
                disabled=not st.session_state.spider_lock)
            
            top_p: float = st.slider(
                label="Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01,
                key="spider_top_p",
                disabled=not st.session_state.spider_lock)
            
            frequency_penalty: float = st.slider(
                label="Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="spider_frequency_penalty",
                disabled=not st.session_state.spider_lock)
            
            presence_penalty: float = st.slider(
                label="Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                key="spider_presence_penalty",
                disabled=not st.session_state.spider_lock)
        
        if st.session_state.spider_chat_history:
            with st.expander("Article Preview"):
                st.markdown(st.session_state.spider_chat_history[0]["content"])
        for i in st.session_state.spider_chat_history[1:]:
            with st.chat_message(i["role"]):
                st.markdown(i["content"])
        
        if query := st.chat_input(
            placeholder="Say something...",
            key="spider_query"):
            if st.session_state.spider_chat_msg == []:
                with st.spinner("Fetching article..."):
                    link_type: str = spider.check_url(query)
                    if link_type == "gzh":
                        text: str = spider.gzh_spider(query)
                    elif link_type == "rmw":
                        text: str = spider.rmw_spider(query)
                    elif link_type == "xhw":
                        text: str = spider.xhw_spider(query)
                    elif link_type == "gmw":
                        text: str = spider.gmw_spider(query)
                    elif link_type == "cnyt":
                        text: str = spider.cnyt_spider(query)
                    elif link_type == "general":
                        text: str = spider.general_spider(query)
                    st.session_state.web_content = text
                    if len(st.session_state.web_content) < 8000:
                        st.session_state.spider_chat_msg.append({"role": "user", "content": st.session_state.web_content})
                        st.session_state.continue_ = True
                        with st.expander("Article Preview"):
                            st.markdown(st.session_state.web_content)
                    else:
                        st.warning(f"Character length: {len(st.session_state.web_content)}. Do you continue?")
                        with st.expander("Article Preview"):
                            st.markdown(st.session_state.web_content)
                        if st.button("Yes", "yes", type="primary"):
                            st.session_state.spider_chat_msg.append({"role": "user", "content": st.session_state.web_content})
                            st.session_state.continue_ = True
                        elif st.button("No", "no"):
                            st.session_state.web_content = ""
                            st.rerun()
            else:
                st.session_state.spider_chat_msg.append({"role": "user", "content": query})
                st.session_state.continue_ = True
                with st.chat_message("user"):
                    st.markdown(query)
            
            if st.session_state.continue_:
                messages: list = [{"role": "system", "content": system_prompt}] + st.session_state.spider_chat_msg
                with st.chat_message("assistant"):
                    client = OpenAI(api_key=api_key, base_url=base_url)
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stream=True)
                    result: str = st.write_stream(chunk.choices[0].delta.content \
                        for chunk in response if chunk.choices[0].delta.content is not None)
                st.session_state.spider_chat_msg.append({"role": "assistant", "content": result})
                st.session_state.spider_chat_history = st.session_state.spider_chat_msg
                st.session_state.continue_ = False
                st.rerun()
        
        if clear_button:
            st.session_state.spider_chat_msg = []
            st.session_state.spider_chat_history = []
            st.session_state.web_content = ""
            st.session_state.spider_sys = spider_prompt
            st.rerun()
        
        if undo_button:
            del st.session_state.spider_chat_msg[-1]
            del st.session_state.spider_chat_history[-1]
            st.rerun()
        
        if retry_button:
            st.session_state.spider_chat_msg.pop()
            st.session_state.spider_chat_history = []
            st.session_state.spider_retry_state = True
            st.rerun()
        if st.session_state.spider_retry_state:
            with st.expander("Article Preview"):
                st.markdown(st.session_state.spider_chat_history[0]["content"])
            for i in st.session_state.spider_chat_msg[1:]:
                with st.chat_message(i["role"]):
                    st.markdown(i["content"])
            messages: list = [{"role": "system", "content": system_prompt}] + st.session_state.spider_chat_msg
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.spider_chat_msg.append({"role": "assistant", "content": result})
            st.session_state.spider_chat_history = st.session_state.spider_chat_msg
            st.session_state.spider_retry_state = False
            st.rerun()
    
    elif func_select == "Translator":
        api_key: str = siliconflow_api_key
        base_url: str = siliconflow_base_url

        max_tokens: int = 4096
        temperature: float = 0.7
        top_p: float = 0.7

        st.subheader("Translator", anchor=False)

        if "translator_sys" not in st.session_state:
            st.session_state.translator_sys = translation_prompt

        if "translator_msg" not in st.session_state:
            st.session_state.translator_msg = []
        if "translator_history" not in st.session_state:
            st.session_state.translator_history = []
        if "translator_retry_state" not in st.session_state:
            st.session_state.translator_retry_state = False
        
        if "translator_lock" not in st.session_state:
            st.session_state.translator_lock = True
        
        if st.session_state.translator_msg != []:
            st.session_state.translator_lock = False
        else:
            st.session_state.traslator_lock = True
        
        with st.sidebar:
            model_list: list = translation_model
            model: str = st.selectbox(
                label="Model",
                options=model_list,
                index=0,
                key="translator_model_select",
                disabled=not st.session_state.translator_lock)

            system_prompt: str = st.text_area(
                label="System Prompt",
                value=st.session_state.translator_sys,
                key="translator_system_prompt",
                disabled=not st.session_state.translator_lock)
            st.session_state.translator_sys = system_prompt
        
        for i in st.session_state.translator_history:
            with st.chat_message(i["role"]):
                st.markdown(i["content"])
        
        if query := st.chat_input("Say something...", key="translator_query"):
            st.session_state.translator_msg.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            messages: list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}]
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.translator_msg.append({"role": "assistant", "content": result})
            st.session_state.translator_history = st.session_state.translator_msg
            st.rerun()
        
        if clear_button:
            st.session_state.translator_msg = []
            st.session_state.translator_history = []
            st.session_state.translator_sys = translation_prompt
            st.rerun()
        
        if undo_button:
            del st.session_state.translator_msg[-1]
            del st.session_state.translator_history[-1]
            st.rerun()
        
        if retry_button:
            st.session_state.translator_msg.pop()
            st.session_state.translator_history = []
            st.session_state.translator_retry_state = True
            st.rerun()
        if st.session_state.translator_retry_state:
            for i in st.session_state.translator_msg:
                with st.chat_message(i["role"]):
                    st.markdown(i["content"])
            messages: list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": st.session_state.translator_msg[0]["content"]}]
            with st.chat_message("assistant"):
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True)
                result: str = st.write_stream(chunk.choices[0].delta.content \
                    for chunk in response if chunk.choices[0].delta.content is not None)
            st.session_state.translator_msg.append({"role": "assistant", "content": result})
            st.session_state.translator_history = st.session_state.translator_msg
            st.session_state.translator_retry_state = False
            st.rerun()