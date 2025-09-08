import json
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import http
from openai import OpenAI
from omegaconf import OmegaConf
import os
from dotenv import load_dotenv

# Register custom OmegaConf resolvers for mathematical operations
def register_math_resolvers():
    """Register custom resolvers for mathematical operations in OmegaConf"""
    try:
        OmegaConf.register_new_resolver("div", lambda x, y: int(float(x) // float(y)))
        OmegaConf.register_new_resolver("mul", lambda x, y: int(float(x) * float(y)))
        OmegaConf.register_new_resolver("add", lambda x, y: int(float(x) + float(y)))
        OmegaConf.register_new_resolver("sub", lambda x, y: int(float(x) - float(y)))
    except Exception:
        pass  # Resolvers might already be registered

# Register the resolvers
register_math_resolvers()


load_dotenv()


# def format_web_search_response(data) -> str:
#     """Format web_search response to show indexed results with URL and Title."""
#     try:
#         if isinstance(data, dict) and 'results' in data and data['results']:
#             formatted_lines = []
#             for idx, result in enumerate(data['results']):
#                 url = result.get('url', 'N/A')
#                 title = result.get('metadata', {}).get('paper_title', 'No title')
#                 if title.endswith("..."):
#                     title = title[:-3]  # Remove trailing "..."
#                 preview = result.get('preview', 'No preview available')
#                 if preview.endswith("..."):
#                     preview = preview[:-3]  # Remove trailing "..."
#                 formatted_lines.append(f"Index: {idx}")
#                 formatted_lines.append(f"URL: {url}")
#                 formatted_lines.append(f"Title: {title}")
#                 formatted_lines.append(f"Preview: {preview}")
#                 if idx < len(data['results']) - 1:
#                     formatted_lines.append("")  # Empty line between results
#             return "\n".join(formatted_lines)
#         else:
#             return str(data)
#     except Exception as e:
#         return f"Error formatting web_search response: {str(e)}"


def format_web_search_response(response_json: str) -> str:
    """Format web_search response to show indexed results with URL and Title."""
    try:
        data = json.loads(response_json)
        if 'results' in data and data['results']:
            formatted_lines = []
            for idx, result in enumerate(data['results']):
                url = result.get('url', 'N/A')
                title = result.get('metadata', {}).get('paper_title', 'No title')
                preview = result.get('preview', 'No preview available')
                formatted_lines.append(f"Index: {idx}")
                formatted_lines.append(f"URL: {url}")
                formatted_lines.append(f"Title: {title}")
                formatted_lines.append(f"Preview: {preview}")
                if idx < len(data['results']) - 1:
                    formatted_lines.append("")  # Empty line between results
            return "\n".join(formatted_lines)
        else:
            return response_json
    except:
        return response_json


# def format_web_visit_response(data) -> str:
#     """Format web_visit response to extract only the 'data' field content."""
#     try:
#         if isinstance(data, dict) and 'data' in data:
#             return data['data']
#         elif isinstance(data, str):
#             return data
#         else:
#             return str(data)
#     except Exception as e:
#         return f"Error formatting web_visit response: {str(e)}"

def format_web_visit_response(response_json: str) -> str:
    """Format web_visit response to extract only the 'data' field content."""
    try:
        data = json.loads(response_json)
        if 'data' in data:
            return data['data']
        else:
            return response_json
    except:
        return response_json

def web_search(query: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Perform a web search to local"""
    try:
        query = query.strip()
        # top_k = config.get("default_num_results", 5)
        top_k = 10
        preview_char = 256
        
        if not query:
            error_msg = "Error: No search query provided"
            return json.dumps({'error': error_msg, 'results': []})
        
        # Get search API URL from config
        search_api_url = "http://127.0.0.1:10000/search"  # default
        search_timeout = 30  # default
        
        if config and 'search_agent' in config:
            search_config = config['search_agent']
            search_api_url = search_config.get('api', {}).get('search_url', search_api_url)
            search_timeout = search_config.get('timeouts', {}).get('search', search_timeout)
        
        # Call external search API
        import pdb; pdb.set_trace()
        client = OpenAI()
        response = requests.post(
            search_api_url,
            json={
                'query': query,
                'top_k': top_k,
                'preview_char': preview_char
            },
            timeout=search_timeout
        )
        response.raise_for_status()
        
        # Return raw JSON response
        return json.dumps(response.json())
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling search API: {str(e)}"
        # logger.error(f"[ERROR] {error_msg}")
        return json.dumps({'error': error_msg, 'results': []})
    except Exception as e:
        error_msg = f"Error during web search: {str(e)}"
        # logger.error(f"Web search execution failed: {e}")
        return json.dumps({'error': error_msg, 'results': []})

# def web_search(query: str, min_results: int = 3, max_retries: int = 3) -> dict:
#     """Perform a web search using Serper API with retry logic to ensure minimum results."""
#     import time
    
#     for attempt in range(max_retries):
#         conn = None
#         try:
#             conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)
#             payload = json.dumps({
#                 "q": query,
#                 "num": 10
#             })
#             headers = {
#                 'Content-Type': 'application/json',
#                 'X-API-KEY': os.getenv('SERPAPI_API_KEY', '')
#             }

#             conn.request("POST", "/search", payload, headers)
#             res = conn.getresponse()
#             raw_data = res.read()
            
#             # Check if response is valid
#             if res.status != 200:
#                 if attempt == max_retries - 1:
#                     return {'results': [], 'error': f"HTTP Error {res.status}: {res.reason}"}
#                 time.sleep(1)  # Wait before retry
#                 continue
            
#             result = json.loads(raw_data.decode("utf-8"))

#             # Format the results
#             formatted_results = {
#                 'results': []
#             }
#             if "organic" in result and result["organic"]:
#                 for item in result["organic"]:
#                     snippet = item.get('snippet', 'No preview available')
                        
#                     formatted_results['results'].append({
#                         'url': item.get("link", ""),
#                         'metadata': {
#                             'paper_title': item.get('title', 'No title')
#                         },
#                         'preview': snippet
#                     })    
            
#             # Check if we have enough results
#             if len(formatted_results['results']) >= min_results:
#                 return formatted_results
#             elif attempt < max_retries - 1:
#                 print(f"Only got {len(formatted_results['results'])} results, retrying... (attempt {attempt + 1}/{max_retries})")
#                 time.sleep(1)  # Wait before retry
#                 continue
#             else:
#                 # Last attempt, return whatever we got
#                 return formatted_results
                
#         except json.JSONDecodeError as e:
#             if attempt == max_retries - 1:
#                 return {'results': [], 'error': f"JSON parsing error: {str(e)}"}
#             time.sleep(1)
#             continue
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 return {'results': [], 'error': f"Error performing web search: {str(e)}"}
#             time.sleep(1)
#             continue
#         finally:
#             if conn:
#                 conn.close()
    
#     return {'results': [], 'error': f"Failed to get {min_results} results after {max_retries} attempts"}
    

# def clean_web_content(content: str) -> str:
#     """Clean web content by removing URLs, extra whitespace, and limiting length."""
#     import re
    
#     if not content:
#         return ""
    
#     # Remove URLs (http/https/ftp/www links)
#     url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*|www\.[^\s<>"{}|\\^`\[\]]*|ftp://[^\s<>"{}|\\^`\[\]]*'
#     content = re.sub(url_pattern, '', content)
    
#     # Remove email addresses
#     email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     content = re.sub(email_pattern, '', content)
    
#     # Remove markdown links [text](url)
#     markdown_link_pattern = r'\[([^\]]*)\]\([^)]*\)'
#     content = re.sub(markdown_link_pattern, r'\1', content)
    
#     # Remove excessive whitespace, newlines, and special characters
#     content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
#     content = re.sub(r'\n+', '\n', content)  # Replace multiple newlines with single newline
#     content = re.sub(r'[\t\r\f\v]+', ' ', content)  # Replace tabs and other whitespace
    
#     # Remove common navigation/footer text patterns
#     noise_patterns = [
#         r'Skip to main content',
#         r'Privacy Policy',
#         r'Terms of Service',
#         r'Cookie Policy',
#         r'© \d{4}',
#         r'All rights reserved',
#         r'Subscribe to newsletter',
#         r'Follow us on',
#         r'Share this',
#         r'Print this page'
#     ]
    
#     for pattern in noise_patterns:
#         content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
#     # Strip leading/trailing whitespace
#     content = content.strip()
    
#     # Cap at 10000 characters
#     if len(content) > 10000:
#         content = content[:10000] + "\n[Content truncated to 10,000 characters]"
    
#     return content

# def web_visit(url: str) -> dict:
#     """Visit a webpage and extract its content using Serper API."""
#     conn = None
#     try:
#         conn = http.client.HTTPSConnection("scrape.serper.dev", timeout=30)
#         payload = json.dumps({
#             "url": url,
#             "includeMarkdown": True
#         })
#         headers = {
#             'Content-Type': 'application/json',
#             'X-API-KEY': os.getenv('SERPAPI_API_KEY', '')
#         }

#         conn.request("POST", "/", payload, headers)
#         res = conn.getresponse()
#         raw_data = res.read()
        
#         # Check if response is valid
#         if res.status != 200:
#             return {'data': f"HTTP Error {res.status}: {res.reason}"}
        
#         result = json.loads(raw_data.decode("utf-8"))

#         # Extract text content
#         raw_content = ""
#         if "markdown" in result and result["markdown"]:
#             raw_content = result["markdown"]
#         elif "text" in result and result["text"]:
#             raw_content = result["text"]
#         else:
#             raw_content = json.dumps(result, indent=2)

#         # Clean and process the content
#         cleaned_content = clean_web_content(raw_content)
        
#         return {'data': cleaned_content}
#     except json.JSONDecodeError as e:
#         return {'data': f"JSON parsing error: {str(e)}"}
#     except Exception as e:
#         return {'data': f"Error visiting webpage: {str(e)}"}
#     finally:
#         if conn:
#             conn.close()


def web_visit(url: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Visit a webpage from local"""

    try:
        url = url.strip()
        
        if not url:
            error_msg = "Error: No URL provided"
            return json.dumps({'error': error_msg, 'content': ''})
        
        # Special handling for Wikipedia URLs
        if 'wiki/' in url:
            url = url.replace('_', '%20')  # Preserve original encoding
        
        # Get visit API URL from config
        visit_api_url = "http://127.0.0.1:10000/visit"  # default
        visit_timeout = 60  # default
        
        if config and 'search_agent' in config:
            search_config = config['search_agent']
            visit_api_url = search_config.get('api', {}).get('visit_url', visit_api_url)
            visit_timeout = search_config.get('timeouts', {}).get('visit', visit_timeout)
        
        # Call external visit API
        response = requests.post(
            visit_api_url,
            json={'url': url},
            timeout=visit_timeout
        )
        response.raise_for_status()
        
        # Return raw JSON response
        return json.dumps(response.json())
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling visit API: {str(e)}"
        # logger.error(f"[ERROR] {error_msg}")
        return json.dumps({'error': error_msg, 'content': ''})
    except Exception as e:
        error_msg = f"Error during web visit: {str(e)}"
        # logger.error(f"Web visit execution failed: {e}")
        return json.dumps({'error': error_msg, 'content': ''})
    

class SearchAgent:
    """
    A search agent that can perform multi-turn searches with reasoning capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: str = "config.yaml"):
        """
        Initialize the search agent.
        
        Args:
            config: Configuration dictionary (if None, loads from config_path)
            config_path: Path to YAML configuration file
        """
        # Load configuration from YAML file if not provided
        if config is None:
            if os.path.exists(config_path):
                config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config = config
        
        # Extract search agent configuration
        search_config = config.get('search_agent', {})
        model_config = search_config.get('model', {})
        
        # Use search agent model config or fall back to main model config
        base_url = model_config.get('base_url') or config.get('model', {}).get('base_url', 'http://localhost:8000/v1')
        api_key = model_config.get('api_key') or config.get('model', {}).get('api_key', 'EMPTY')
        model_name = model_config.get('model_name', 'zai-org/GLM-4.5-FP8')
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.conversation_history = []
        self.search_history = []
        self.web_search_count = 0
        self.web_visit_count = 0
        
        self.max_turns = search_config.get('max_turns', 10)
        self.default_num_results = search_config.get('default_num_results', 3)
        
        # Calculate per_turn_max_tokens dynamically if not set
        max_tokens = config.get('model', {}).get('sampling', {}).get('max_tokens', 40960)
        self.per_turn_max_tokens = search_config.get('per_turn_max_tokens', max_tokens // self.max_turns)
        
        self.system_prompt = search_config.get('system_prompt', """
You are a helpful assistant. You are answering a question in medical/ healthcare domain. Please perform multi-turn searches, visits and provide comprehensive answers

When answering questions, you should:
1. First THINK about what information you need to gather
2. Use search to find relevant information
3. Use visit to get detailed content from specific URLs
4. Reason through the information systematically
5. Provide a comprehensive final answer

Always use your reasoning capabilities to plan your search strategy and synthesize information effectively. 

Final note:
1. If user provide you the output format that they expect, please strictly follow it.
2. Users want you to answer with your knowledge, therefore you don't want to let them know that you cheated by searching the information on internet. Since users are only allowed to see your final answer, never say something like: "Based on my research..." or "Now I have enough information.." in that final response.
        """.strip())
        
        # Define available functions
        self.functions = [
            {
                "type": "function",  # This wrapper is required
                "function": {        # Function details go here
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",  # Each tool needs this wrapper
                "function": {
                    "name": "web_visit",
                    "description": "Visit a URL and retrieve the full document content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to visit"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
        
        # Create partial functions with config
        from functools import partial
        self.available_functions = {
            'web_search': partial(web_search, config=self.config.search_agent),
            'web_visit': partial(web_visit, config=self.config)
        }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        self.search_history = []
        self.web_search_count = 0
        self.web_visit_count = 0
    
    def _add_system_prompt(self):
        """Add a system prompt to guide the agent's behavior."""
        return [{'role': 'system', 'content': self.system_prompt}]
    
    def _extract_reasoning_from_response(self, response) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content and regular content from GLM-4.5 response.
        
        Returns:
            tuple: (reasoning_content, regular_content)
        """
        reasoning_content = None
        regular_content = None
        
        # Handle OpenAI API response format
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            # GLM-4.5 provides reasoning_content as a separate field
            if hasattr(message, 'reasoning_content'):
                reasoning_content = message.reasoning_content
            
            # Extract regular content
            if hasattr(message, 'content'):
                regular_content = message.content
            
            # Alternative: Check if response object has reasoning_content directly
            if not reasoning_content and hasattr(choice, 'reasoning_content'):
                reasoning_content = choice.reasoning_content
            
            # For some API providers, reasoning might be in the message dict
            if not reasoning_content and isinstance(message, dict):
                reasoning_content = message.get('reasoning_content')
                if not regular_content:
                    regular_content = message.get('content')
        
        # Handle dictionary response format (some providers return dict instead of object)
        elif isinstance(response, dict) and 'choices' in response:
            choices = response.get('choices', [])
            if choices:
                choice = choices[0]
                message = choice.get('message', {})
                
                # Extract reasoning_content and content from dictionary
                reasoning_content = message.get('reasoning_content')
                regular_content = message.get('content')
        
        return reasoning_content, regular_content
    
    def _parse_tool_calls(self, response) -> List[Dict[str, Any]]:
        """
        Parse tool calls from GLM-4.5 response.
        
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            # Standard OpenAI tool_calls format
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        'id': tool_call.id,
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        },
                        'type': 'function'
                    })
        
        return tool_calls

    def search(self, user_query: str, max_turns: Optional[int] = None) -> str:
        """
        Perform a multi-turn search based on user query.
        
        Args:
            user_query: The user's search query
            max_turns: Maximum number of conversation turns (uses config default if None)
            
        Returns:
            Final response from the agent
        """
        # Use config max_turns if not specified
        if max_turns is None:
            max_turns = self.max_turns
            
        # Initialize conversation with system prompt and user query
        messages = self._add_system_prompt()
        # user_query += '\n\nPlease reason step-by-step, and put your final answer within \\boxed{}.'
        messages.append({'role': 'user', 'content': user_query})
        self.conversation_history = messages.copy()
        
        # print(f"\n{'='*60}")
        # print(f"User Query: {user_query}")
        # print(f"{'='*60}\n")
        
        final_response = ""
        turn = 0
        sampling_config = self.config.model.sampling
        extra_body = {"chat_template_kwargs": {"enable_thinking": self.config.model.enable_thinking}}
        if "Qwen3" in self.model_name:
            extra_body.update({"top_k": 20, "min_p": 0.0,})
        
        while True:
            turn += 1
            # print(f"\n--- Turn {turn} ---")
            # Get response from LLM (non-streaming)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.functions,
                tool_choice="auto",
                max_tokens=self.per_turn_max_tokens,
                temperature=sampling_config.temperature,
                top_p=sampling_config.top_p,
                extra_body=extra_body,
                stream=False,
            )
            if not response:
                break
            # Extract reasoning and content
            reasoning_content, regular_content = self._extract_reasoning_from_response(response)

            # if reasoning_content:
            #     print(f"\n[Reasoning/Thinking Process]:")
            #     print(reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content)

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    
                    # print(f"\n[Tool Call]: {function_name}")
                    # print(f"Arguments: {json.dumps(function_args, indent=2)}")
                    
                    # Execute the function
                    function_to_call = self.available_functions.get(function_name)
                    if function_to_call:
                        if function_name == 'web_search':
                            function_response = function_to_call(
                                query=function_args.get('query')
                            )
                            self.web_search_count += 1
                            formatted_response = format_web_search_response(function_response)
                            self.search_history.append({
                                'type': 'search',
                                'query': function_args.get('query'),
                                'timestamp': datetime.now().isoformat()
                            })
                        elif function_name == 'web_visit':
                            function_response = function_to_call(
                                url=function_args.get('url')
                            )
                            self.web_visit_count += 1
                            formatted_response = format_web_visit_response(function_response)
                            self.search_history.append({
                                'type': 'visit',
                                'url': function_args.get('url'),
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Add tool response to messages
                        messages.append({
                            'role': 'assistant',
                            'content': regular_content if regular_content else f"Calling {function_name}...",
                            'tool_calls': [tool_call]
                        })
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call['id'],
                            'name': function_name,
                            'content': formatted_response
                        })
                        
                        # print(f"\n[Tool Response Preview]: {formatted_response[:200]}...")
            else:
                # No tool calls - this is likely the final response
                if regular_content:
                    # print(f"\n[Final Response]:")
                    # print(regular_content)
                    final_response = regular_content
                    
                    # Add to messages for conversation history
                    messages.append({
                        'role': 'assistant',
                        'content': regular_content
                    })
                    break
            
            # Safety check: prevent infinite loops
            if turn > max_turns:
                # print("\n[Stopping after 50 turns to prevent infinite loop]")
                break
                
            # Check for repeated responses (same content 2 times in a row) - only for assistant messages
            assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get('role') == 'assistant']
            if len(assistant_messages) >= 2:
                last_assistant_content = assistant_messages[-1].get('content', '')
                prev_assistant_content = assistant_messages[-2].get('content', '')
                if (last_assistant_content and prev_assistant_content and 
                    last_assistant_content == prev_assistant_content and 
                    len(last_assistant_content) > 100):
                    # print("\n[Stopping - detected repeated assistant response]")
                    break
        
        messages[-1]["search_history"] = self.search_history
        messages[-1]["usage"] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens}
        self.conversation_history = messages
        return final_response
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the history of searches and visits performed."""
        return self.search_history


def interactive_search(config_path: str = "config.yaml"):
    """Run an interactive search session."""
    agent = SearchAgent(config_path=config_path)
    
    # Get commands from config
    interactive_commands = agent.config.get('search_agent', {}).get('interactive_commands', {
        'exit': 'Quit the program',
        'history': 'View search history',
        'reset': 'Clear conversation history'
    })
    
    commands_str = ', '.join([f"'{cmd}' ({desc})" for cmd, desc in interactive_commands.items()])
    print(f"Search Agent initialized. Commands: {commands_str}")
    
    while True:
        user_input = input("\nEnter your search query: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'history':
            history = agent.get_search_history()
            print("\nSearch History:")
            for item in history:
                print(f"  - {item['type']}: {item.get('query', item.get('url'))} at {item['timestamp']}")
        elif user_input.lower() == 'reset':
            agent.reset_conversation()
            print("Conversation reset.")
        else:
            response = agent.search(user_input)
            print(f"\n{'='*60}")
            print("Final Answer:")
            print(f"{'='*60}")
            print(response)


if __name__ == '__main__':
    # Example usage
    agent = SearchAgent()
    
    # Example 1: Simple search
    # print("Example 1: Simple search")
    # response = agent.search("What are the latest developments in quantum computing?")
    # print(f"\nFinal response:\n{response}")
    
    # # Example 2: Complex multi-step search
    # print("\n\nExample 2: Complex search requiring multiple steps")
    # # agent.reset_conversation()
    # response = agent.search("What should a user do if they are under 18 years of age and want to use nicotine lozenges?")
    # print(f"\nFinal response:\n{response}")
    print("\n\nStarting interactive mode...")
    interactive_search()
