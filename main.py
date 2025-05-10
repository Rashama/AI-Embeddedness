import json
from openai import OpenAI
from tools.sentiment_tool import sentiment_tool, analyze_sentiment
from tools.multimodal_tool import multimodal_tool, analyze_multimodal_content
from tools.gemini_tool import gemini_tool, process_with_gemini
from utils.logger import enhanced_logger
from config.config import OPENAI_API_KEY
import os
from typing import Dict, List,Any
from datetime import datetime

# tool_call_logger = ToolCallLogger()

class AIAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.tools = [sentiment_tool, multimodal_tool, gemini_tool]
        self.conversation_history = []
        self.current_file_path = None
        self.logger = enhanced_logger

    def set_file_path(self, file_path: str):
        if file_path and os.path.exists(file_path):
            self.current_file_path = file_path
            # logger.info(f"File path set to: {file_path}")
            return True
        return False

    def add_to_history(self, role: str, content: str, tool_results: Dict = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if tool_results:
            message["tool_results"] = tool_results
        
        self.conversation_history.append(message)
        # logger.info(f"Added to history - Role: {role}, Content: {content}")

    def _get_system_prompt(self, file_path: str = None) -> str:
        """Get the system prompt for the conversation"""
        return f"""You are a helpful AI assistant that can:
                1. Provide basic image classification only (using analyze_multimodal_content)
                2. Translate text to french (using analyze_multimodal_content with translation parameters)
                3. Analyze sentiment (using analyze_sentiment)
                4. Process content with Gemini for advance tasks related to image,pdf,videos or other tasks which is not handled by above tool below are its capability 
                - Advanced text and sentiment analysis
                - Advance image and video processing 
                - Document understanding and extraction
                - Multi-language translation
                - Content generation
                (using process_with_gemini)

                Current file path: {file_path if file_path else 'No file'}

                When processing queries:
                1. Consider the previous results when they're relevant
                2. For translations to french only, use the analyze_multimodal_content tool with appropriate language parameters
                3. For new image analysis, always use the current file path
                4. Chain operations logically when multiple steps are needed.
                5. Only use tools to answer queries do not use your own knowledge.
        """

    def process_query(self, user_input: str) -> str:
        try:
            # Parse file path if present
            file_path = None
            query = user_input

            if '|' in user_input:
                parts = user_input.split('|')
                for part in parts:
                    if 'file:' in part.lower():
                        file_path = part.split('file:')[1].strip()
                    if 'query:' in part.lower():
                        query = part.split('query:')[1].strip()

            # Use class file path if set
            if not file_path:
                file_path = self.current_file_path

            # Create conversation ID for tracking
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initial system message
            messages = [{
                "role": "system",
                "content": self._get_system_prompt(file_path)
            }, {
                "role": "user",
                "content": query
            }]

            # Get tool selection response
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            tool_results = []
            message = response.choices[0].message

            if not message.tool_calls:
                # Log direct response
                self.logger.log_conversation(
                    user_query=query,
                    file_path=file_path,
                    final_response=message.content,
                    conversation_id=conversation_id
                )
                return message.content

            # Process tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if file_path:
                    function_args['file_path'] = file_path

                try:
                    # Execute tool
                    result = self._execute_tool(function_name, function_args)
                    
                    tool_results.append({
                        "tool_name": function_name,
                        "arguments": function_args,
                        "result": result
                    })

                except Exception as e:
                    error = f"Error in {function_name}: {str(e)}"
                    tool_results.append({
                        "tool_name": function_name,
                        "error": error
                    })

            # Process the tool results using GPT-4
            processed_response = self._process_tool_results(query, tool_results)

            # Log the conversation
            self.logger.log_conversation(
                user_query=query,
                file_path=file_path,
                tool_name=", ".join(t["tool_name"] for t in tool_results),
                tool_args=function_args,
                tool_response=tool_results,
                final_response=processed_response,
                conversation_id=conversation_id
            )

            return processed_response

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.logger.error(error_msg, exc_info=True)
            return error_msg
        
    def _process_tool_results(self, original_query: str, tool_results: List[Dict]) -> str:
        """Process tool results using GPT-4 to generate a human-friendly response"""
        try:
            # Format tool results for GPT
            tool_results_str = json.dumps(tool_results, indent=2)
            
            # Create prompt for processing results
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that interprets tool results and provides clear, 
                    concise explanations. Format your response in a natural, easy-to-understand way. Focus on the key 
                    information and insights from the tool results.Keep explanation related to original query only do not include your knowledge"""
                },
                {
                    "role": "user",
                    "content": f"""Original query: {original_query}
                    
                    Tool results:
                    {tool_results_str}
                    
                    Please provide a clear, natural response that addresses the original query using these tool results. 
                    Explain any insights or findings in a conversational way."""
                }
            ]

            # Get GPT's interpretation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error processing tool results: {str(e)}\nRaw results: {json.dumps(tool_results, indent=2)}"


    def add_to_history(self, role: str, content: str, tool_results: Dict = None):
        """Add message and tool results to conversation history"""
        message = {
            "role": role,
            "content": content
        }
        if tool_results:
            message["tool_results"] = tool_results
            
        self.conversation_history.append(message)
        # logger.info(f"Added to history - Role: {role}, Content: {content}")

    def set_file_path(self, file_path: str):
        """Set the current file path and verify it exists"""
        if file_path and os.path.exists(file_path):
            self.current_file_path = file_path
            # logger.info(f"File path set to: {file_path}")
            return True
        return False

    def _execute_tool(self, function_name: str, function_args: Dict) -> Any:
        """Execute a specific tool with given arguments"""
        if function_name == "analyze_sentiment":
            return analyze_sentiment(**function_args)
        elif function_name == "analyze_multimodal_content":
            return analyze_multimodal_content(**function_args)
        elif function_name == "process_with_gemini":
            return process_with_gemini(**function_args)
        raise ValueError(f"Unknown tool: {function_name}")

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool results into a presentable response"""
        if len(tool_results) == 1:
            result = tool_results[0].get("result")
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            return str(result)
        return json.dumps(tool_results, indent=2)

def main():
    """Main function for terminal interface"""
    agent = AIAgent()
    print("AI Agent initialized. Type 'exit' to quit.")
    print("For file processing, use format: 'file: path_to_file | query: your_query'")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                break
            response = agent.process_query(user_input)
            print("\nAssistant:", response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            # logger.error(f"Error in main loop: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()