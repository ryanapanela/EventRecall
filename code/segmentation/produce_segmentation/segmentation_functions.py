# %% Import Modules
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, zscore
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI
import backoff
import tqdm
import re
from typing import List
import torch
from transformers import pipeline

# %% Set OpenAI API Key
API_KEY = 'API-KEY'
client = OpenAI(api_key=API_KEY)

# %% Initialize LLaMA 3 Model
pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# %% Helper Functions
def clean_text(text_path):
	"""
	Cleans the text content of a file by removing specific characters and replacing others.

	Args:
		text_path (str): The file path to the text file to be cleaned.

	Returns:
		str: The cleaned text content of the file.

	Notes:
		- Removes double quotes (").
		- Removes left and right double quotation marks (Unicode: \u201C, \u201D).
		- Replaces ellipses (...) with a single period (.).
		- Replaces hyphens (-) with spaces.
	"""
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
	"""
	Cleans and formats the output string from a language model (LLM) by applying
	a series of regular expression substitutions. The function performs the following:

	1. Ensures double newlines between sections by replacing single newlines with double newlines.
	2. Removes numbered list prefixes (e.g., "1.", "2.", etc.).
	3. Removes event markers (e.g., "Event 1:", "Event 2:", etc.).
	4. Removes segment markers (e.g., "Segment 1: ", "Segment 2: ", etc.).
	5. Cleans up extra spaces around double newlines.
	6. Strips leading whitespace from the entire string.

	Args:
		output (str): The raw output string from the LLM.

	Returns:
		str: The cleaned and formatted output string.
	"""
	temp_output = re.sub('(?<=\\n)\\s(?=\\n)', '\\n\\n', output)
	temp_output = re.sub('(?<!\\n)\\n(?!\\n)', '\\n\\n', temp_output)
	temp_output = re.sub(r"[0-9]+\.", "", temp_output)
	temp_output = re.sub(r"Event \d+:", "", temp_output)
	temp_output = re.sub(r"Segment \d+: ", "", temp_output)
	temp_output = re.sub("(?<=\\n\\n)\\s", "", temp_output)
	temp_output = re.sub(" (?=\\n\\n)", "", temp_output)
	temp_output = re.sub("^\\s", "", temp_output)
	temp_output = temp_output.lstrip()

	return temp_output


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def prompt_gpt(model, message, temperature=0, max_tokens=4096, frequency_penalty=0, presence_penalty=0):
    """
    Sends a chat completion request to the specified GPT model with the given parameters.

    Args:
        model (str): The name of the model to use for generating the response.
        message (str): The input message or prompt to send to the model.
        temperature (float, optional): Controls the randomness of the response. 
            Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0) make it more deterministic. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens to include in the response. Defaults to 4096.
        frequency_penalty (float, optional): Penalizes new tokens based on their frequency in the text so far. 
            A higher value reduces the likelihood of repeating the same phrases. Defaults to 0.
        presence_penalty (float, optional): Penalizes new tokens based on whether they appear in the text so far. 
            A higher value increases the likelihood of introducing new topics. Defaults to 0.

    Returns:
        dict: The response from the model, typically containing the generated text and other metadata.
    """
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
	"""
	Sends a chat completion request to the specified LLaMA model with the given parameters.

	Args:
		message (str): The input message or prompt to be sent to the model.
		temperature (float, optional): Sampling temperature for controlling randomness in the output. 
			Lower values make the output more deterministic. Defaults to 0.1.
		max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.
		frequency_penalty (float, optional): Penalty for repeated tokens in the output. 
			Higher values reduce repetition. Defaults to 0.

	Returns:
		str: The generated response from the LLaMA model.
	"""
	prompt_message = [{"role": "user", "content": message}]
	response = pipe(prompt_message,
				 max_new_tokens=max_tokens,
				 temperature=temperature,
				 repetition_penalty=(1+frequency_penalty))

	return response

def get_output(responses, choice: int = 0):
    """
    Extracts and returns the content of a specific choice from the given responses object.

    Args:
        responses: An object containing a list of choices, where each choice may have a 
                   'message.content' or 'text' attribute.
        choice (int, optional): The index of the choice to extract. Defaults to 0.

    Returns:
        str: The content of the specified choice, either from 'message.content' or 'text'.

    Raises:
        IndexError: If the specified choice index is out of range.
        AttributeError: If the expected attributes ('choices', 'message.content', or 'text') 
                        are not present in the responses object.
    """
    try:
        return responses.choices[choice].message.content

    except:
        return responses.choices[choice].text


def get_finish_reason(responses, choice=0):
	"""
	Retrieve the finish reason for a specific choice from the responses object.

	Args:
		responses (object): The object containing the choices and their metadata.
		choice (int): The index of the choice to retrieve the finish reason for.

	Returns:
		str: The finish reason associated with the specified choice.
	"""
	return responses.choices[choice].finish_reason


def split_text(parsed_text):
	"""
	Splits a list of strings into sublists of segmented events based on double newline delimiters.

	Args:
		parsed_text (list of str): A list of strings where each string may contain multiple events 
									separated by double newline characters ('\n\n').

	Returns:
		list of list of str: A list where each element is a sublist containing the segmented events 
								from the corresponding string in the input list.
	"""
	segmented_events = []
	for i in parsed_text:
		segmented_events.append(i.split('\n\n'))

	return segmented_events

# %% GPT Segmentation Function
def gpt_segmentation(text_path, iters=1, model='gpt-4', temperature=0):
	"""
	Segments a story into large coherent events using a specified GPT model.

	Args:
		text_path (str): The file path to the text that needs to be segmented.
		iters (int, optional): The number of iterations to run the segmentation process. Defaults to 1.
		model (str, optional): The name of the language model to use for segmentation. Defaults to 'gpt-4'.
		temperature (float, optional): Controls the randomness of the model's output. Defaults to 0.

	Returns:
		list: A list of segmented events derived from the input text.

	Notes:
		- The function reads and cleans the input text, constructs a prompt for the language model,
			and processes the model's responses to extract segmented events.
		- Only responses with a 'stop' finish reason are considered for segmentation.
		- The function assumes the presence of helper functions such as `clean_text`, `prompt_gpt`, 
			`parse_llm_output`, `get_output`, `get_finish_reason`, and `split_text`.
	"""
	text = clean_text(text_path)

	prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into \
	large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins. \
	This is the story: "

	prompt_offset = "\n This is a word-for-word copy of the same story that is segmented into large event units: "

	responses = []
	iter_times = iters

	for i in tqdm.tqdm(range(iter_times)):
		curr_prompt = prompt_onset + text + prompt_offset

		curr_response = prompt_gpt(model, curr_prompt, temperature=temperature)

		# Save the current model
		responses.append(curr_response)

	parsed_LLM = [parse_llm_output(get_output(item)) for item in responses]
	finished_reasons = [get_finish_reason(item) for item in responses]

	parsed_LLM_stop = []
	for i in range(len(parsed_LLM)):
		if finished_reasons[i] == 'stop':
			parsed_LLM_stop.append(parsed_LLM[i])

	segmented_events = split_text(parsed_LLM)

	return segmented_events

# %% LLaMA Segmentation Function
def llama_segmentation(text_path, iters=1, temperature=0.1):
	"""
	Segments a story into large coherent events using LLaMA.

	Args:
		text_path (str): The file path to the text that needs to be segmented.
		iters (int, optional): The number of iterations to run the segmentation process. Defaults to 1.
        temperature (float, optional): Controls the randomness of the model's output. Defaults to 0.

	Returns:
		list: A list of segmented events derived from the input text.

	Notes:
		- The function reads and cleans the input text, constructs a prompt for the language model,
			and processes the model's responses to extract segmented events.
		- Only responses with a 'stop' finish reason are considered for segmentation.
		- The function assumes the presence of helper functions such as `clean_text`, `prompt_llama`, 
			`parse_llm_output`, `get_output`, `get_finish_reason`, and `split_text`.
	"""
	# Ensure temperature is not zero
	if temperature == 0:
		temperature = 0.1

	text = clean_text(text_path)

	prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into \
	large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins. \
	This is the story: "

	prompt_offset = "\n This is a word-for-word copy of the same story that is segmented into large event units: "

	responses = []
	iter_times = iters

	for i in tqdm.tqdm(range(iter_times)):
		curr_prompt = prompt_onset + text + prompt_offset
		curr_response = prompt_llama(curr_prompt, temperature=temperature)

		# Save the current model
		responses.append(curr_response)

	parsed_LLM = [parse_llm_output(get_output(item)) for item in responses]
	finished_reasons = [get_finish_reason(item) for item in responses]

	parsed_LLM_stop = []
	for i in range(len(parsed_LLM)):
		if finished_reasons[i] == 'stop':
			parsed_LLM_stop.append(parsed_LLM[i])

	segmented_events = split_text(parsed_LLM)

	return segmented_events