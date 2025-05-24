import json

async def get_summary(history, turn_count, bedrock):
    """Generate a summary of the conversation."""
    if turn_count >= 5:
        text_summary = ""
        for turn in history:
            user_statement = turn["user"]
            aria_statement = turn["titan"]
            text_summary += user_statement + aria_statement + "\n"

        summary_system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Please condense our conversation into a summary. Include a list of main "
            "points or key takeaways and highlight the most important ideas or insights. "
            "Finish with any concluding thoughts or next steps. The user will prompt you "
            "with a multi-turn conversation between and AI and a user. Please summarize "
            "it using the denoted guidance and reply with nothing more than a generated "
            "summary. Thank you. <|eot_id|>"
        )
        summary_user_prompt = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{text_summary}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        summary_prompt = summary_system_prompt + summary_user_prompt

        model_kwargs = json.dumps(
            {"prompt": summary_prompt,
             "temperature": 0.5,
             "top_p": 0.9,
             "max_gen_len": 2048}
        )
        response = bedrock.invoke_model(
            body = model_kwargs,
            modelId = "us.meta.llama3-2-11b-instruct-v1:0"
        )
        response_body = json.loads(response.get('body').read())

        summarized_conversation = response_body['generation']

        summary_result = (
            "<|start_header_id|>summary<|end_header_id|>\n"
            f"{summarized_conversation}"
            "<|eot_id|>"
        )

        return summary_result

    return "<|start_header_id|>summary<|end_header_id|>\n<|eot_id|>"

async def get_history(history, turn_count):
    """Generate a history of the conversation."""
    text_history = ""
    #history.append(f"Turn {turn_count}:\nUser: {user_input}\nAria: {aria_response}\n")
    #print(turn_count)
    if turn_count >= 5:
        history.pop(0)

    for turn in history:
        text_history += (f"Turn {turn["turn"]}:\nUser: {turn["user"]}\nTitan: {turn["titan"]}\n")
    history_prompt = "<|start_header_id|>history<|end_header_id|>\n" + text_history + "<|eot_id|>"
    return history_prompt

async def get_user(user_input):
    """Generate a user prompt."""
    user_prompt = (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    return user_prompt

def chat_titan(bedrock, sys_prompt, summary_prompt, history_prompt, user_prompt):
    """Generate a streaming response from Titan."""
    print("\033[0;31m" + sys_prompt)
    print("\033[1;32m" + summary_prompt)
    print("\033[1;34m" + history_prompt)
    print("\033[1;33m" + user_prompt)

    full_prompt = sys_prompt + summary_prompt + history_prompt + user_prompt
    model_kwargs = json.dumps({
        "prompt": full_prompt,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_gen_len": 2048
    })

    response = bedrock.invoke_model_with_response_stream(
        body=model_kwargs,
        modelId="us.meta.llama3-2-90b-instruct-v1:0"
    )

    # Use the provided callback or default to process_chunk
    full_response = ""

    try:
        event_stream = response.get('body')
        for event in event_stream:
            chunk = json.loads(event['chunk']['bytes'].decode())
            chunk_text = chunk['generation']
            full_response += chunk_text
            yield chunk_text

    except Exception as e:
        print(f"Error processing stream: {e}")
