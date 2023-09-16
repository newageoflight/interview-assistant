from agents.interviewee import make_interviewee

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import streamlit as sl

from utils.pdf_loader import pdf_to_kbase

load_dotenv()

"""
This chatbot is based on a prompt I gave to the addon-enabled ChatGPT-4 on the web, but I've programmed it with the additional ability
to read your CV and search Google to give grounded advice.
"""

# TODO: improve the quality of the output somehow, it honestly kinda sucks
# TODO: allow the AI to ingest data from the BPT networks' websites, slides, etc.
# should be able to form a knowledge base on this somehow e.g. crawling RACP website, job descriptions
# TODO: allow AI to ingest user's CV in PDF format and a personal description
# cf. this: https://medium.com/@johnthuo/chat-with-your-pdf-using-langchain-f-a-i-s-s-and-openai-to-query-pdfs-e7bfde086155


def main():
    sl.title("BPT Interview Assistant")
    # TODO: fiddle around with the settings a bit, higher temps seem to do better
    chat = ChatOpenAI(model="gpt-4", temperature=0.9)

    # TODO: Replace with a CV upload dialog
    cv_pdf = sl.file_uploader("Upload your CV in PDF format", type="pdf")
    if cv_pdf is not None:
        cv_kbase = pdf_to_kbase(cv_pdf)
        retriever = cv_kbase.as_retriever(return_source_documents=True)
        qa_chain = RetrievalQA.from_chain_type(
            chat, chain_type="stuff", retriever=retriever, verbose=True)
        agent = make_interviewee(chat, qa_chain)
        query = sl.text_input("What question would you like to ask?")
        cancel_button = sl.button("Cancel")

        if cancel_button:
            sl.stop()

        if query:
            with get_openai_callback() as cost:
                res = agent.run(query)
                print(cost)

            sl.write(res)


if __name__ == "__main__":
    main()
