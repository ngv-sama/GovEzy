import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai
import streamlit as st
import streamlit.components.v1 as components
import re
from streamlit_option_menu import option_menu
openai.api_key = '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'
import streamlit_mermaid as stmd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from streamlit_ace import st_ace, KEYBINDINGS, THEMES, LANGUAGES
from gtts import gTTS  # new import
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import time
import streamlit_mermaid as stmd
import random
from pydub import AudioSegment
import tempfile
import os
import base64
from IPython.display import HTML

st.set_page_config(layout="wide")
selected=option_menu(
    menu_title=None,
    options=["Visualize",  "Upload", "News"],
    icons=["diagram-3", "upload", "newspaper"],
    default_index=0,
    orientation="horizontal",
)

# if selected=="News":
#         import streamlit as st
#         import requests

#         # Replace with your own key
#         API_KEY = 'b3d7dc02aab146cfa2e688a3453c3751'

#         def get_news(api_key, query):
#             url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
#             response = requests.get(url)
#             return response.json()

#     # Streamlit app

#         st.title('Keep in touoch with the news that matters to you!!!')
        
#         query = st.text_input('Enter a topic to search for', 'Python')
#         if st.button('Get News'):
#             news_json = get_news(API_KEY, query)
#             articles = news_json.get('articles', [])
            
#             for article in articles:
#                 st.subheader(article['title'])
#                 st.write(article['description'])
#                 st.markdown(f"[Read More]({article['url']})")

# if selected=="News":
#     import streamlit as st
#     import requests
#     from bs4 import BeautifulSoup
#     from fpdf import FPDF
#     import io

#     # Replace with your own key
#     API_KEY = 'b3d7dc02aab146cfa2e688a3453c3751'

#     def get_news(api_key, query):
#         url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
#         response = requests.get(url)
#         return response.json()

#     def scrape_article(url):
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         text = ''
#         for paragraph in soup.find_all('p'):
#             text += paragraph.text
#         return text

#     def create_pdf(content, title):
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size = 12)
#         pdf.cell(200, 10, txt = title, ln = True, align = 'C')
#         pdf.multi_cell(0, 10, txt = content)
        
#         pdf_output = io.BytesIO()
#         pdf.output(pdf_output, 'F')
#         pdf_output.seek(0)
#         return pdf_output

#     # Streamlit app
#     st.title('Keep in touch with the news that matters to you!!!')

#     query = st.text_input('Enter a topic to search for', 'Python')
#     if st.button('Get News'):
#         news_json = get_news(API_KEY, query)
#         articles = news_json.get('articles', [])
        
#         for article in articles:
#             st.subheader(article['title'])
#             st.write(article['description'])
#             article_url = article['url']
#             st.markdown(f"[Read More]({article['url']})")
#             if st.button(f"Get PDF for {article['title']}"):
#                 content = scrape_article(article_url)
#                 pdf_file = create_pdf(content, article['title'])
#                 st.download_button(label="Download PDF",
#                                 data=pdf_file,
#                                 file_name=f"{article['title']}.pdf",
#                                 mime='application/pdf')


# if selected=="News":
#     import streamlit as st
#     import requests
#     from bs4 import BeautifulSoup
#     from fpdf import FPDF
#     import io

#     # Replace with your own key
#     API_KEY = 'b3d7dc02aab146cfa2e688a3453c3751'

#     def get_news(api_key, query):
#         url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
#         response = requests.get(url)
#         return response.json()

#     def scrape_article(url):
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         text = ''
#         for paragraph in soup.find_all('p'):
#             text += paragraph.text
#         return text

#     def create_pdf(content, title):
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size = 12)
#         pdf.cell(200, 10, txt = title, ln = True, align = 'C')
#         pdf.multi_cell(0, 10, txt = content)
        
#         pdf_output = io.BytesIO()
#         pdf.output(pdf_output, 'F')
#         pdf_output.seek(0)
#         return pdf_output

#     # Streamlit app
#     st.title('Keep in touch with the news that matters to you!!!')

#     query = st.text_input('Enter a topic to search for', 'Python')
#     if st.button('Get News'):
#         news_json = get_news(API_KEY, query)
#         articles = news_json.get('articles', [])
#         for i, article in enumerate(articles):
#             st.subheader(article['title'])
#             st.write(article['description'])
#             st.markdown(f"[Read More]({article['url']})")

#             # Unique button for each article
#             if st.button(f"Get PDF for {article['title']}", key=f"button_{i}"):
#                 st.session_state['url_to_scrape'] = article['url']
#                 st.session_state['title_to_scrape'] = article['title']

#     # Check if an article URL is stored in session state
#     if 'url_to_scrape' in st.session_state and 'title_to_scrape' in st.session_state:
#         content = scrape_article(st.session_state['url_to_scrape'])
#         pdf_file = create_pdf(content, st.session_state['title_to_scrape'])
#         st.download_button(label="Download PDF",
#                         data=pdf_file,
#                         file_name=f"{st.session_state['title_to_scrape']}.pdf",
#                         mime='application/pdf')
#         # Clear the session state after downloading
#         del st.session_state['url_to_scrape']
#         del st.session_state['title_to_scrape']

if selected=="News":
    import streamlit as st
    import requests
    from bs4 import BeautifulSoup
    from fpdf import FPDF
    import io

    # Replace with your own key
    API_KEY = 'b3d7dc02aab146cfa2e688a3453c3751'

    def get_news(api_key, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        response = requests.get(url)
        return response.json()

    def scrape_article(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ''
        for paragraph in soup.find_all('p'):
            text += paragraph.text
        return text

    def create_pdf(content, title):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)
        pdf.cell(200, 10, txt = title, ln = True, align = 'C')
        pdf.multi_cell(0, 10, txt = content)
        
        pdf_output = io.BytesIO()
        pdf.output(pdf_output, 'F')
        pdf_output.seek(0)
        return pdf_output

    # Streamlit app
    st.title('Keep in touch with the news that matters to you!!!')

    query = st.text_input('Enter a topic to search for', 'Python')
    if st.button('Get News'):
        news_json = get_news(API_KEY, query)
        articles = news_json.get('articles', [])
        
        for article in articles:
            with st.expander(f"{article['title']}"):
                st.write(article['description'])
                st.markdown(f"[Read More]({article['url']})")
                # if st.button(f"Get PDF for {article['title']}", key=article['title']):
                #     content = scrape_article(article['url'])
                #     pdf_file = create_pdf(content, article['title'])
                    # st.download_button(label="Download PDF",
                    #                 data=pdf_file,
                    #                 file_name=f"{article['title']}.pdf",
                    #                 mime='application/pdf')






def mermaid_ele(code: str) -> None:
    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=800,
    )

def text_to_speech(text):
    """
    Converts text to an audio file using gTTS and returns the audio file as binary data
    """
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()

if selected=="Visualize":
    st.title("Gov Ezy")
    st.write("Say hello to Gov Ezy, that helps make policy easy, quick and reachable!!!")

    # Get prompt from user
    prompt = st.text_input("Visuzlize your policy in an easy manner:")

    # On prompt submission, send request to OpenAI API
    if st.button("Making it Ezy"):
        with st.spinner('Analyzing...'):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"""
                You are a teacher and your student has asked you a question: {prompt}. 
                Please provide a detailed theoretical answer in bullet points. 
                You must always give an answer.
                Also, create a Mermaid.js diagram to visually represent the information. 
                The diagram should be formatted as code and contain at least 15 nodes. 
                It should be top to bottom orientation. 
                Feel free to add as many nodes as necessary and cycles if needed. 
                Here's an example of a Mermaid.js diagram for reference:
                flowchart TD 
                A[Christmas] -->|Get money| B[Go shopping]
                B --> C[Let me think]
                C -->|One| D[Laptop]
                C -->|Two| E[iPhone]
                C -->|Three| F[Car]
                The diagram should be oriented from top to bottom and use labels extensively. 
                After viewing the diagram, the student should have no further questions.
                Please start the Mermaid.js code with ‘MERMAID_START’ and end it with ‘MERMAID_END’. 
                The diagram should be the last part of the answer, not inserted in the middle."
                """,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            col1, col2 = st.columns(2)
            text_body=""
            
        
            with col1:
                # Extract mermaid code from response
                mermaid_code = response.choices[0]['text']
                text_data=mermaid_code.split('MERMAID_START')
                # print(text_data)
                corpus=text_data[0].split('\n')
                for i in corpus:
                    text_body+=i
                    st.write(i.strip())
                st.audio(text_to_speech(text_body), format="audio/wav")
                # video_paths = ['metahuman_1.mp4', 'metahuman_2.mp4', 'metahuman_3.mp4']
                
                # selected_video_path = random.choice(video_paths)

                # # Generate speech audio
                # speech_audio = AudioSegment.from_file(BytesIO(text_to_speech(text_body)), format="mp3")

                # # Display the video and audio using HTML
                # video_html = f'<video controls autoplay loop width="100%"><source src="data:video/mp4;base64,{base64.b64encode(open(selected_video_path, "rb").read()).decode()}" type="video/mp4"></video>'
                # audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{base64.b64encode(speech_audio.export(format="mp3").read()).decode()}" type="audio/mp3"></audio>'
                # display_html = f"{video_html}<br>{audio_html}"
                # st.markdown(display_html, unsafe_allow_html=True)
            
            with col2:
                start_marker = 'MERMAID_START'
                end_marker = 'MERMAID_END'
                start_index = response.choices[0].text.find(start_marker)
                end_index = response.choices[0].text.find(end_marker)
                if start_index != -1 and end_index != -1:
                    mermaid = response.choices[0].text[start_index + len(start_marker):end_index].strip()
                else:
                    mermaid = "No mermaid.js graph found in the response."
                
                mermaid_ele(mermaid)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


if selected=="Upload":
    load_dotenv()
    # st.set_page_config(page_title="Chat with multiple PDFs",
    #                    page_icon=":books:")
    # st.title("Chat with multiple PDFs :books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)



