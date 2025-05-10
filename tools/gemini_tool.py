import google.generativeai as genai
import PIL.Image
import time
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass
from config.config import GEMINI_API_KEY

@dataclass
class ContentTool:
    name: str
    description: str
    supported_types: List[str]
    process_func: callable

class UnifiedGeminiAgent:
    """
    A unified agent that automatically handles different types of content through tools.
    """
    
    def __init__(self, model_id: str = "gemini-1.5-flash"):
        """Initialize the agent with API key and tools."""
        self.model_id = model_id
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.model_id)
        
        # Initialize tools
        self._initialize_tools()
        
        # Initialize chat for interactive sessions
        self.chat = self.model.start_chat()

    def _initialize_tools(self):
        """Initialize content processing tools."""
        self.tools = {
            'text': ContentTool(
                name="text_processor",
                description="Process text-only content",
                supported_types=['text/plain'],
                process_func=self._process_text
            ),
            'image': ContentTool(
                name="image_processor",
                description="Process image content",
                supported_types=['image/jpeg', 'image/png', 'image/gif'],
                process_func=self._process_image
            ),
            'audio': ContentTool(
                name="audio_processor",
                description="Process audio content",
                supported_types=['audio/mpeg', 'audio/wav', 'audio/x-wav'],
                process_func=self._process_audio
            ),
            'video': ContentTool(
                name="video_processor",
                description="Process video content",
                supported_types=['video/mp4', 'video/mpeg', 'video/quicktime'],
                process_func=self._process_video
            ),
            'document': ContentTool(
                name="document_processor",
                description="Process document content",
                supported_types=['application/pdf'],
                process_func=self._process_document
            )
        }

    def _get_content_type(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Determine content type from file or assume text if no file provided."""
        if file_path is None:
            return 'text/plain'
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'

    def _get_appropriate_tool(self, content_type: str) -> Optional[ContentTool]:
        """Get the appropriate tool for the content type."""
        for tool in self.tools.values():
            if content_type in tool.supported_types:
                return tool
        return None

    def _process_text(self, prompt: str, **kwargs) -> str:
        """Process text-only content."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error processing text: {str(e)}"

    def _process_image(self, prompt: str, file_path: Union[str, Path], **kwargs) -> str:
        """Process image content."""
        try:
            image = PIL.Image.open(file_path)
            response = self.model.generate_content([prompt, image])
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def _process_audio(self, prompt: str, file_path: Union[str, Path], **kwargs) -> str:
        """Process audio content."""
        try:
            audio_file = genai.upload_file(file_path)
            response = self.model.generate_content([prompt, audio_file])
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error processing audio: {str(e)}"

    def _process_video(self, prompt: str, file_path: Union[str, Path], **kwargs) -> str:
        """Process video content."""
        try:
            video_file = genai.upload_file(file_path)
            
            # Wait for video processing
            while video_file.state.name == "PROCESSING":
                print("Processing video...")
                time.sleep(5)
                video_file = genai.get_file(video_file.name)
            
            response = self.model.generate_content([prompt, video_file])
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error processing video: {str(e)}"

    def _process_document(self, prompt: str, file_path: Union[str, Path], **kwargs) -> str:
        """Process document content."""
        try:
            doc_file = genai.upload_file(file_path)
            response = self.model.generate_content([prompt, doc_file])
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def process(self, 
                prompt: str, 
                file_path: Optional[Union[str, Path]] = None, 
                **kwargs) -> str:
        """
        Main processing method that automatically handles different content types.
        
        Args:
            prompt (str): The prompt or query for the model
            file_path (Optional[Union[str, Path]]): Path to the file to process (if any)
            **kwargs: Additional arguments for specific tools
            
        Returns:
            str: Generated response
            
        Raises:
            ValueError: If content type is not supported
        """
        try:
            content_type = self._get_content_type(file_path)
            tool = self._get_appropriate_tool(content_type)
            
            if tool is None:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Process the content using the appropriate tool
            if file_path is None:
                return tool.process_func(prompt, **kwargs)
            else:
                return tool.process_func(prompt, file_path, **kwargs)
        except Exception as e:
            return f"Error in process: {str(e)}"

# Create a singleton instance
gemini_agent = UnifiedGeminiAgent()

# Tool definition for OpenAI function calling
gemini_tool = {
    "type": "function",
    "function": {
        "name": "process_with_gemini",
        "description": "Useful for advance question answering Advanced text and sentiment analysis, Advance image and video processing ,Document understanding and extraction, Multi-language translation,Content generation",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt or query to process."
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to be processed (image, audio, video, or document). if available"
                },
                "file_type": {
                    "type": "string",
                    "description": "Type of file (text, image, audio, video, document)",
                    "enum": ["text", "image", "audio", "video", "document"]
                }
            },
            "required": ["prompt"]
        }
    }
}

def process_with_gemini(prompt: str, file_path: str = None, file_type: str = None) -> str:
    """
    Process content using Gemini model.
    
    Args:
        prompt (str): The prompt or query to process
        file_path (str, optional): Path to the file to process
        file_type (str, optional): Type of file for explicit type declaration
        
    Returns:
        str: Generated response from Gemini
    """
    return gemini_agent.process(prompt=prompt, file_path=file_path)