# Functions for automated event segmentation

import os
import pandas as pd
import numpy as np
import re
import tqdm
import openai
from openai import OpenAI
import backoff
from transformers import pipeline
from typing import List

def clean_text(text_path):
    text_file = open(text_path, 'r', encoding='utf-8-sig')
    text = text_file.read()
    text_file.close()
    text = text.replace('"', '')
    text = text.replace('\u201C', '')
    text = text.replace('\u201D', '')
    text = text.replace('...', '.')
    text = text.replace('-', ' ')
    return text

def parse_llm_output(output: str):
    temp_output = re.sub('(?<=\n)\s(?=\n)', '\n\n', output)
    temp_output = re.sub('(?<!\n)\n(?!\n)', '\n\n', temp_output)
    temp_output = re.sub(r"[0-9]+\.", "", temp_output)
    temp_output = re.sub(r"Event \d+:", "", temp_output)
    temp_output = re.sub(r"Segment \d+: ", "", temp_output)
    temp_output = re.sub("(?<=\n\n)\s", "", temp_output)
    temp_output = re.sub(" (?=\n\n)", "", temp_output)
    temp_output = re.sub("^\s", "", temp_output)
    temp_output = temp_output.lstrip()
    return temp_output

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def prompt_gpt(api_key, model, message, temperature=0, max_completion_tokens=4096, frequency_penalty=0, presence_penalty=0):
    client = OpenAI(api_key=api_key)
    prompt_message = [{"role": "user", "content": message}]
    curr_response = client.chat.completions.create(
        model=model,
        messages=prompt_message,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return curr_response

def prompt_llama(message, temperature=0.1, max_tokens=4096, frequency_penalty=0):
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
    prompt_message = [{"role": "user", "content": message}]
    response = pipe(prompt_message,
                 max_new_tokens=max_tokens,
                 temperature=temperature,
                 repetition_penalty=(1+frequency_penalty))
    return response

def get_output(responses, choice: int = 0):
    try:
        return responses.choices[choice].message.content
    except:
        return responses.choices[choice].text

def get_finish_reason(responses, choice=0):
    return responses.choices[choice].finish_reason

def split_text(parsed_text):
    segmented_events = []
    for i in parsed_text:
        segmented_events.append(i.split('\n\n'))
    return segmented_events

def gpt_segmentation(text_path, api_key, iters=1, model='gpt-4', temperature=0):
    text = clean_text(text_path)
    prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins. This is the story: "
    prompt_offset = "\n This is a word-for-word copy of the same story that is segmented into large event units: "
    responses = []
    for i in tqdm.tqdm(range(iters)):
        curr_prompt = prompt_onset + text + prompt_offset
        curr_response = prompt_gpt(api_key, model, curr_prompt, temperature=temperature)
        responses.append(curr_response)
    parsed_LLM = [parse_llm_output(get_output(item)) for item in responses]
    finished_reasons = [get_finish_reason(item) for item in responses]
    parsed_LLM_stop = [parsed_LLM[i] for i in range(len(parsed_LLM)) if finished_reasons[i] == 'stop']
    segmented_events = split_text(parsed_LLM_stop)

    return segmented_events

def llama_segmentation(text_path, iters=1, temperature=0.1):
    if temperature == 0:
        temperature = 0.1
    text = clean_text(text_path)
    prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins. This is the story: "
    prompt_offset = "\n This is a word-for-word copy of the same story that is segmented into large event units: "
    responses = []
    for i in tqdm.tqdm(range(iters)):
        curr_prompt = prompt_onset + text + prompt_offset
        curr_response = prompt_llama(curr_prompt, temperature=temperature)
        responses.append(curr_response)
    parsed_LLM = [parse_llm_output(get_output(item)) for item in responses]
    finished_reasons = [get_finish_reason(item) for item in responses]
    parsed_LLM_stop = [parsed_LLM[i] for i in range(len(parsed_LLM)) if finished_reasons[i] == 'stop']
    segmented_events = split_text(parsed_LLM_stop)
    
    return segmented_events

def run_segmentation(input_path, api_key, model='gpt-4', iters=1, temperature=0):
    """
    Run event segmentation on the input file using LLMs.
    Args:
        input_path (str): Path to input file (CSV or TXT)
        api_key (str): OpenAI API key
        output_dir (str): Directory to save results
        model (str): Model to use ('gpt-4' or 'llama')
        iters (int): Number of iterations
        temperature (float): Sampling temperature
    Returns:
        segmented_events (list): List of segmented events
    """
    if model == 'llama':
        segmented_events = llama_segmentation(input_path, iters=iters, temperature=temperature)
    else:
        segmented_events = gpt_segmentation(input_path, api_key, iters=iters, model=model, temperature=temperature)

    return segmented_events

def find_event_boundaries(word_num: List[int]) -> List[int]:
	"""Given a list of word numbers representing the lengths of events, 
    return a list of integers with the associated word numbers at event boundaries.

    Args:
        word_num (list[int]): List of word numbers associated with events.

    Returns:
        list[int]: List of word numbers at event boundary locations.
	"""
	boundaries = []

	for i in range(1, len(word_num)):
		# Cumulative word count up to the current event and 
		# add 1 because event boundary is located at successive word
		word_count = sum(word_num[:i]) + 1  
		if word_count != sum(word_num[:i - 1]):
			boundaries.append(word_count)

	return boundaries

def event_data(events: list[str]) -> list[int]:
	""" Given a list of events, return a list of integers with the associated 
	word numbers.

	Args:
		events (list[str]): List of events from gpt_segmentation output

	Return:
		list[int]: List of word numbers associated with event boundary location
	"""

	length_events = [len(event.split()) for event in events]
	word_boundaries = find_event_boundaries(length_events)

	return word_boundaries

def segmentation(input_path: str, api_key: str, output_path: str = None, model: str = 'gpt-4', iters: int = 1, temperature: float = 0) -> None:
    """Run segmentation on the input file and save results.

    Args:
        input_path (str): Path to input file (CSV or TXT)
        api_key (str): OpenAI API key
        output_dir (str): Directory to save results
        model (str): Model to use ('gpt-4' or 'llama')
        iters (int): Number of iterations
        temperature (float): Sampling temperature
    """
    segmentations = run_segmentation(input_path, api_key, model=model, iters=iters, temperature=temperature)
    
    rows = []
    for iteration_idx, events in enumerate(segmentations, start=1):
        word_boundaries = event_data(events)
        for event_idx, (event, word_num) in enumerate(zip(events, word_boundaries), start=1):
            rows.append({
                'input_file': os.path.basename(input_path),
                'iteration': iteration_idx,
                'temperature': temperature,
                'model': model,
                'event_number': event_idx,
                'event': event,
                'word_number': word_num
            })

    output_file = pd.DataFrame(rows)
    if output_path is not None:
         output_file.to_csv(output_path, index=False)

    return output_file, segmentations