import sys
import json

# from unichat import UnifiedChatApi, MODELS_LIST
import unichat


def validate_inputs(api_key: str, model_name: str) -> None:
    """Validate the API key and model name."""
    if not api_key:
        raise ValueError("API key cannot be empty")
    if not any(model_name in models_list for models_list in unichat.MODELS_LIST.values()):
        raise ValueError(f"Unsupported model: {model_name}")


def get_calculation(tool_call):
    """Process calculator tool calls."""
    args_str = tool_call['function']['arguments']
    args = json.loads(args_str)

    def calculate(operation, operand1, operand2):
        if operation == "add":
            return operand1 + operand2
        elif operation == "subtract":
            return operand1 - operand2
        elif operation == "multiply":
            return operand1 * operand2
        elif operation == "divide":
            if operand2 == 0:
                raise ValueError("Cannot divide by zero.")
            return operand1 / operand2
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    result = calculate(
        args['operation'],
        float(args['operand1']),
        float(args['operand2'])
    )

    return {
        "role": "tool",
        "content": str(result),
        "tool_call_id": tool_call['id']
    }


def handle_streaming_response(response_stream, conversation):
    """Handle streaming response and tool calls."""

    current_content = ""
    current_assistant_message = {
        "role": "assistant",
        "content": "",
    }

    # Flags to track if labels have been printed
    printed_reasoning_label = False
    printed_assistant_label = False

    last_tool_call_index = -1

    for chunk in response_stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        # Handle reasoning_content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not printed_reasoning_label:
                print("\nAssistant Reasoning: ", end="")
                printed_reasoning_label = True
            print(delta.reasoning_content, end="")

        # Handle content
        if hasattr(delta, 'content') and delta.content:
            if not printed_assistant_label:
                if printed_reasoning_label:
                    print("\n", end="")  # Add newline to separate from reasoning
                print("\nAssistant: ", end="")
                printed_assistant_label = True
            print(delta.content, end="")
            current_content += delta.content
            current_assistant_message['content'] = current_content

        # Handle tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            # Initialize tool_calls if not present
            if 'tool_calls' not in current_assistant_message:
                current_assistant_message['tool_calls'] = []

            for tool_call in delta.tool_calls:
                # If we have an ID, this is a new tool call
                if hasattr(tool_call, 'id') and tool_call.id:
                    new_tool_call = {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name if hasattr(tool_call.function, 'name') else "",
                            'arguments': ""
                        }
                    }
                    current_assistant_message['tool_calls'].append(new_tool_call)
                    last_tool_call_index = len(current_assistant_message['tool_calls']) - 1

                # If we have arguments, append to the last tool call
                if (hasattr(tool_call, 'function') and
                    hasattr(tool_call.function, 'arguments') and
                    tool_call.function.arguments):
                    if last_tool_call_index >= 0:
                        current_assistant_message['tool_calls'][last_tool_call_index]['function']['arguments'] += tool_call.function.arguments

        if finish_reason:
            conversation.append(current_assistant_message)

            # Process tool calls
            if current_assistant_message.get('tool_calls'):
                for tool_call in current_assistant_message['tool_calls']:
                    if tool_call['function']['name'] == "calculator":
                        try:
                            result = get_calculation(tool_call)
                            conversation.append(result)
                        except Exception as e:
                            raise

    print()


def handle_non_streaming_response(response, conversation):
    """Handle non-streaming response and tool calls."""
    assistant_response = {
        "role": "assistant",
        "content": "",
    }

    message = response.choices[0].message

    # Safely get tool_calls
    tool_calls = getattr(message, 'tool_calls', None)

    # Handle reasoning content
    reasoning_content = getattr(message, 'reasoning_content', None)
    if reasoning_content:
        assistant_response['reasoning_content'] = reasoning_content
        print("\nAssistant Reasoning: ", reasoning_content)

    # Handle content
    content = getattr(message, 'content', None)
    if content:
        assistant_response['content'] = content
        print("\nAssistant: ", content)

    # Handle tool calls if they exist
    if tool_calls:
        assistant_response['tool_calls'] = []
        for tool_call in tool_calls:
            # Safely get attributes using getattr
            tool_id = getattr(tool_call, 'id', '')
            tool_fn = getattr(tool_call, 'function', None)
            if tool_fn:
                tool_name = getattr(tool_fn, 'name', '')
                tool_args = getattr(tool_fn, 'arguments', '{}')

                new_tool_call = {
                    'id': tool_id,
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'arguments': tool_args
                    }
                }
                assistant_response['tool_calls'].append(new_tool_call)

    # Add the assistant's response to the conversation
    conversation.append(assistant_response)

    # Process tool calls if they exist
    if tool_calls:
        for tool in assistant_response.get('tool_calls', []):
            if tool['function']['name'] == "calculator":
                try:
                    result = get_calculation(tool)
                    conversation.append(result)
                except Exception as e:
                    print(f"Error in calculator: {e}")
                    raise


def main():
    try:
        # Prompt for and validate inputs
        api_key = input("Enter your API key: ").strip()
        model_name = input("Enter the model name (e.g., 'gpt-4o-mini'): ").strip()
        streaming = not input("Type anything to disable streaming or ENTER to continue: ").strip()

        validate_inputs(api_key, model_name)

        # Initialize the client
        client = unichat.UnifiedChatApi(api_key=api_key)

        # Define available tools
        tools = [{
            "name": "calculator",
            "description": "A simple calculator that performs basic arithmetic operations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform."
                    },
                    "operand1": {
                        "type": "number",
                        "description": "The first operand."
                    },
                    "operand2": {
                        "type": "number",
                        "description": "The second operand."
                    }
                },
                "required": ["operation", "operand1", "operand2"]
            }
        }]

        # Set up system message
        system = input("Enter system instructions or leave blank for default: ").strip()
        if not system:
            system = "You are a helpful assistant."

        # Initialize conversation
        conversation = [
            {"role": "system", "content": system}
        ]

        # Start the chat loop
        while True:
            try:
                # Check if we're not in the middle of a tool call
                if conversation[-1]['role'] != "tool":
                    user_message = input("\nYou: ").strip()

                    if not user_message:
                        continue

                    if user_message.lower() in {"exit", "quit"}:
                        print("Exiting the chat.")
                        sys.exit(0)

                    conversation.append({"role": "user", "content": user_message})

                # Get chat completion
                if streaming:
                    response_stream = client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        tools=tools,
                        reasoning_effort = "medium"
                    )
                    handle_streaming_response(response_stream, conversation)
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        tools=tools,
                        stream=False,
                        reasoning_effort = "medium"
                    )
                    handle_non_streaming_response(response, conversation)

            except Exception as e:
                print(f"An error occurred during chat: {e}")
                break

    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()