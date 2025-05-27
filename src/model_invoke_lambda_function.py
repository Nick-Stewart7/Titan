import json
import boto3
 
def lambda_handler(event, context):
    bedrock = boto3.client(service_name='bedrock-runtime')
    sys_prompt = ("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
        You are a troubleshooting assistant focused on AWS cloud services and operating systems. Your role is to help diagnose and solve technical problems while being direct about any limitations in your knowledge.
        Core Principles:

        Express uncertainty when appropriate
        Focus on one issue at a time
        Consider security implications in all solutions

        When troubleshooting:

        Ask for relevant context you need
        Share your reasoning process
        Present solutions as suggestions rather than absolutes
        Note potential risks or side effects
        Admit when you need more information

        Security Guidelines:

        Never request credentials or sensitive data
        Default to least-privilege approaches
        Suggest checking internal security policies

        Remember:

        You can say "I'm not sure" or "I need more information"
        Real solutions often require iteration and testing
        Sometimes the best answer is recommending human expertise
        Not every problem has an immediate solution

        Focus on being helpful while remaining honest about limitations. Adapt your communication style to the user's technical level and the complexity of the problem.<|eot_id|>
        """)
    user_prompt = event['user_prompt']
    """Generate a response from Titan."""
    #print("\033[1;32m" + summary_prompt)
    #print("\033[1;34m" + history_prompt)
    print(f"user_prompt: {user_prompt}")
    user_prompt = f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    full_prompt = sys_prompt + user_prompt
    model_kwargs = json.dumps({
        "prompt": full_prompt,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_gen_len": 2048
    })
    response = bedrock.invoke_model(
        body=model_kwargs,
        modelId="us.meta.llama3-2-90b-instruct-v1:0"
    )
    response_body = json.loads(response.get('body').read())
    print(response_body['generation'])
    return {
        'statusCode': 200,
        'body': json.dumps(response_body['generation'])
    }