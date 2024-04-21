#!/bin/env python

from transformers import pipeline
from argparse import ArgumentParser
import logging
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, JSONLoader, CSVLoader, TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings

LOGGER = logging.getLogger("default")


def get_doc(path: str) -> list:
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path, extract_images=True)
    elif path.endswith('.csv'):
        loader = CSVLoader(path)
    elif path.endswith('.json'):
        loader = JSONLoader(path)
    elif path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(path)
    else:
        loader = TextLoader(path)
    return loader.load()


def search_doc(query: str, database):

    return database.similarity_search(query, k=1)


def parse_doc(segments: list):

    faiss_index = FAISS.from_documents(segments, GPT4AllEmbeddings())
    return faiss_index


TEST_PATH = "~/Documents/contractRedHat.pdf"
TEST_QUERY = "salary"


def main():

    # parser = ArgumentParser(
    #     prog="retrieve",
    #     description="Retrieve info form arbitrary text files using LLM")

    # parser.add_argument('query', type=str)
    # parser.add_argument('--document', '-d', type=str, default=None)

    # # https://huggingface.co/Falconsai/text_summarization Apache 2.0 license
    # parser.add_argument('--model', '-m',
    #                     type=str, default='Falconsai/text_summarization',
    #                     help='Hugging face signature of your chosen model')
    # parser.add_argument('-v', '--verbose', action='store_true')

    # args = parser.parse_args()

    # document = ""

    # if args.verbose:
    #     LOGGER.setLevel(logging.DEBUG)
    # try:
    #     if not args.document:
    #         document = '\n'.join(sys.stdin.readlines())
    #     else:
    #         document = get_doc(args.document)
    # except UnicodeDecodeError:
    #     LOGGER.error("Document decoding failed. Check your encoding!",
    #                  exc_info=True)
    #     exit(1)
    document = get_doc(TEST_PATH)
    database = parse_doc(document)

    results = search_doc(TEST_QUERY, database=database)
    summarizer = pipeline('summarization', model='Falconsai/text_summarization')
    for r in results:
        print(summarizer(r.page_content)[0]['summary_text'])
        #print(str(r.metadata["page"]) + ":", r.page_content)

if __name__ == '__main__':
    main()
