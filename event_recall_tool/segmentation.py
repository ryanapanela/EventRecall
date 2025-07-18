# segmentation.py
# Functions for automated event segmentation

import os
import re
import tqdm
import openai
from openai import OpenAI
import backoff
from transformers import pipeline

def clean_text(text_path):
    # ...existing code from segmentation_functions.py...
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
    # ...existing code from segmentation_functions.py...
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
def prompt_gpt(api_key, model, message, temperature=0, max_tokens=4096, frequency_penalty=0, presence_penalty=0):
    client = OpenAI(api_key=api_key)
    prompt_message = [{"role": "user", "content": message}]
    curr_response = client.chat.completions.create(
        model=model,
        messages=prompt_message,
        temperature=temperature,
        max_tokens=max_tokens,
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
    prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into     large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins.     This is the story: "
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
    prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into     large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins.     This is the story: "
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

def run_segmentation(input_path, api_key, output_dir, model='gpt-4', iters=1, temperature=0):
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
        None (saves results to output_dir)
    """
    if model == 'gpt-4':
        segmented_events = gpt_segmentation(input_path, api_key, iters=iters, model=model, temperature=temperature)
    elif model == 'llama':
        segmented_events = llama_segmentation(input_path, iters=iters, temperature=temperature)
    else:
        raise ValueError('Model must be "gpt-4" or "llama"')
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'segmentation_{model}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for seg in segmented_events:
            f.write("\n---\n".join([s for s in seg if s]) + "\n")
    print(f"Segmentation results saved to {out_path}")
