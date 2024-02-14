#!/bin/env python

from transformers import pipeline, AutoTokenizer
from argparse import ArgumentParser
import os
import logging
import textwrap
import sys
import pypdf

LOGGER = logging.getLogger("default")

def summarize_text(summarizer, text, split_size = 300, min_tokens=200, max_context=None, max_summary_size=1000):
  if not max_context: # Auto set max_context from tokenizer property
    max_context = summarizer.tokenizer.max_len_single_sentence

  ntokens = len(summarizer.tokenizer(text)['input_ids'])
  if ntokens > max_context: # If we can't fit in context split into smaller chunks
    LOGGER.info(f"max context {max_context} exceeded, going deeper!")
    text = text.split()
    samples = [summarize_text(summarizer, ' '.join(text[i:i+split_size]), split_size-10)
      for i in range(0, len(text), split_size)]
    if len(samples) > max_summary_size:
      return summarize_text(summarizer, ' '.join(samples), split_size)
    return ' '.join(samples)
  elif ntokens < min_tokens:
    return text
  else:
    return summarizer(text)[0]['summary_text']

def main():

    parser = ArgumentParser(
        prog="summarize",
        description="Summarize arbitrary text files using LLM")

    parser.add_argument('--document', '-d', type=str, default=None)

    # https://huggingface.co/Falconsai/text_summarization Apache 2.0 license
    parser.add_argument('--model', '-m',
                        type=str, default='Falconsai/text_summarization',
                        help='Hugging face signature of your chosen model')
    parser.add_argument('--max-size', '-S',
                        type=int, default=1000,
                        help='Maximum size of summarized document')
    parser.add_argument('--summary-width', '-C', type=int, default=80,
                        help='Width of summary output in columns')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    document = ""

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    summarizer = pipeline('summarization', model=args.model)
    try:
        if not args.document:
            document = '\n'.join(sys.stdin.readlines())
        else:
            if args.document.endswith('.pdf'):
                reader = pypdf.PdfReader(args.document)
                for page in reader.pages:
                    document += f"\n {page.extract_text()}"
            else:
                with open(args.document, 'r', encoding='utf-8') as f:
                    document = f.read()
    except UnicodeDecodeError:
        LOGGER.error("Document decoding failed. Check your encoding!", exc_info=True)
        exit(1)

    doc_lines = document.split('\n')

    LOGGER.info(f"Original -> lines: {len(doc_lines)} words: {len(document.split(' '))} chars {len(document)}")

    summary = summarize_text(summarizer, document, max_summary_size=args.max_size)

    lines = summary.split('\n')
    LOGGER.info(f"Summarized -> lines: {len(lines)} words: {len(summary.split())} chars {len(summary)}")
    summary = '\n'.join(textwrap.wrap(summary, width=80))

    print(summary)

if __name__ == '__main__':
    main()
