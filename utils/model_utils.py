import better_profanity
import random
import os
import torch
from IPython.display import Markdown
from langchain.schema import Document
from langchain import PromptTemplate
import gc
import torch
from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
import copy
import json
import transformers
transformers.logging.set_verbosity_error()


def _get_summarization_pipeline():
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        skip_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        use_fast=True,
    )
    
    # load the model
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        device_map={"": 0},  # this will load the model in GPU
        torch_dtype=torch.float32,
        return_dict=True,
        load_in_4bit=True
    )
    
    # set up pipeline
    flan_pipeline = pipeline(
        model=model,
        task="summarization",
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        tokenizer=tokenizer,
        num_beams=4,
        min_length=50,
        max_length=150,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )
    return pipeline



def _static_mode_llm_api(prompt, flan_pipeline=None, **kwargs) -> str:
    """ Create a custom API that matches Guardrail.ai requirements, based on an instantiated llm pipeline."""
    # this needs to match the name tag in the RAIL string
    dict_text = {"summarize_statement": flan_pipeline(prompt)[0]["summary_text"]}
    json_text = json.dumps(dict_text)
        
    return json_text



def _my_llm_api(prompt: str, **kwargs) -> str:
    """
    Function to create custom API that matches Guardrail.ai requirements.
    """

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        skip_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        use_fast=True,
    )
    
    # load the model
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        device_map={"": 0},  # this will load the model in GPU
        torch_dtype=torch.float32,
        return_dict=True,
        load_in_4bit=True
    )
    
    # set up pipeline
    flan_pipeline = pipeline(
        model=model,
        task="summarization",
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        tokenizer=tokenizer,
        num_beams=4,
        min_length=50,
        max_length=150,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )

    # this needs to match the name tag in the RAIL string
    dict_text = {"summarize_statement": flan_pipeline(prompt)[0]["summary_text"]}
    json_text = json.dumps(dict_text)
    
    del tokenizer, model, flan_pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    return json_text


def _shortcut_start():
    """
    Function that loads data, model and tokenizer.
    """

    # load dataset
    data = load_from_disk("summaries_dataset")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        skip_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        use_fast=True,
    )

    return data, tokenizer


def _format_llm_output(text):
    """
    Function to apply formatting to the output from the LLMs.
    """
    return Markdown('<div class="alert alert-block alert-info">{}</div>'.format(text))


def _generate_summary(prompt, model, tokenizer):
    """
    Function to invoke model.
    """

    # encode text (tokenize)
    encoded_tokens = tokenizer(prompt, return_tensors="pt", truncation=True)

    # generate summary
    generated_tokens = model.generate(
        encoded_tokens.input_ids.to("cuda"),
        num_return_sequences=1,
        do_sample=False,
        early_stopping=True,
        num_beams=4,
        min_length=50,
        max_length=350,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )

    # garbage collect
    del encoded_tokens
    torch.cuda.empty_cache()

    # convert back
    output_text = tokenizer.decode(generated_tokens.reshape(-1))

    # garbage collect
    del generated_tokens
    torch.cuda.empty_cache()

    return output_text


def _add_summaries(sample, chain):
    """
    Function to create summaries of the movie dialogue dataset.
    """

    # turn off verbosity for chain
    chain.llm_chain.verbose = False

    # create LangChain document from the chunks
    docs = [
        Document(page_content=split["text"], metadata=split["metadata"])
        for split in sample["chunks"]
    ]

    # parse documents through the map reduce chain
    full_output = chain({"input_documents": docs})

    # extract the summary
    summary = full_output["output_text"]

    # return the new column
    sample["summary"] = summary

    # delete objects that are no longer in use
    del docs, summary

    # garbage collect
    gc.collect()

    return sample
