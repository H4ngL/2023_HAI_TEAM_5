"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

llm = OpenAI(model_name='gpt-4', temperature=1.0)
chain = VectorDBQAWithSourcesChain.from_llm(llm, vectorstore=store)

# Streamlit UI
st.set_page_config(page_title="HAI Notion QA Bot", page_icon=":robot:")
st.header("HAI Notion QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Create a form to get user input
with st.form(key="user_input_form"):
    user_input = st.text_area("You: ", "", key="input", max_chars=2000)
    submit_button = st.form_submit_button("Submit")

# Process user input when the submit button is pressed
if submit_button:
    result = chain({"question": user_input + "한글로 답해줘"})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display the chat messages
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        # message = st.chat_message("assistant")
        # message.write("dd")
        # message = st.chat_message("user")
        # message.write("dd")
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")