import streamlit as st
import os
from main import AIAgent
from tempfile import NamedTemporaryFile
import json
from utils.logger import enhanced_logger
from datetime import datetime

def initialize_session_state():
    if 'agent' not in st.session_state:
        st.session_state.agent = AIAgent()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def save_uploaded_file(uploaded_file):
    try:
        if uploaded_file is not None:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
    return None

def display_tool_call(tool_call):
    try:
        # Format timestamp for display
        timestamp = tool_call.get('timestamp', '').split('T')[-1][:8]
        
        with st.expander(f"🛠️ Tool Used: {tool_call['tool_name']} at {timestamp}", expanded=False):
            # Display arguments
            st.markdown("**Arguments:**")
            args_display = tool_call['arguments'].copy()
            if 'file_path' in args_display:
                args_display['file_path'] = f"...{os.path.basename(args_display['file_path'])}"
            st.json(args_display)
            
            # Display results
            st.markdown("**Results:**")
            result = tool_call['result']
            if isinstance(result, str) and len(result) > 500:
                st.markdown(f"{result[:500]}...")
                with st.expander("Show full result"):
                    st.markdown(result)
            else:
                st.json(result)
    except Exception as e:
        st.error(f"Error displaying tool call: {str(e)}")

# def update_tool_monitor():
#     """Update the tool monitor display"""
#     st.markdown("### 🔍 Tool Call Monitor")
#     tool_calls = enhanced_logger.read_tool_calls(limit=5)
    
#     if tool_calls:
#         for tool_call in tool_calls:
#             display_tool_call(tool_call)
#     else:
#         st.info("No tool calls recorded yet")

def display_conversation_history(conversations):
    """Display conversation history with tool details"""
    st.markdown("### 💬 Recent Conversations")
    
    if not conversations:
        st.info("No conversations recorded yet")
        return
        
    for conv in conversations:
        # Format timestamp
        try:
            timestamp = conv['timestamp'].split('.')[0].replace('T', ' ')
        except:
            timestamp = str(conv['timestamp'])
            
        # Create expander header with timestamp
        header = f"🗣️ {timestamp} - {conv['user_query'][:50]}..."
        with st.expander(header, expanded=False):
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Query Details
                st.markdown("**Query Details:**")
                st.markdown(f"- **Query:** {conv['user_query']}")
                
                # Safely handle file path display
                if conv.get('file_path') and isinstance(conv['file_path'], str):
                    try:
                        st.markdown(f"- **File:** {os.path.basename(conv['file_path'])}")
                    except:
                        st.markdown("- **File:** Not available")
                
                if conv.get('tool_name'):
                    st.markdown(f"- **Tool Used:** {conv['tool_name']}")
            
            with col2:
                # Tool Arguments
                if conv.get('tool_arguments'):
                    st.markdown("**Tool Arguments:**")
                    try:
                        if isinstance(conv['tool_arguments'], str):
                            args = json.loads(conv['tool_arguments'])
                            if isinstance(args, dict) and 'file_path' in args:
                                args['file_path'] = f"...{os.path.basename(args['file_path'])}"
                            st.json(args)
                        else:
                            st.text(str(conv['tool_arguments']))
                    except:
                        st.text(str(conv['tool_arguments']))
            
            # Response section
            st.markdown("---")
            st.markdown("**Tool Response:**")
            if conv.get('tool_response'):
                try:
                    if isinstance(conv['tool_response'], str):
                        response = json.loads(conv['tool_response'])
                        if isinstance(response, dict):
                            for key, value in response.items():
                                st.markdown(f"*{key}:*")
                                st.json(value)
                        else:
                            st.json(response)
                    else:
                        st.text(str(conv['tool_response']))
                except:
                    st.text(str(conv['tool_response']))
            
            if conv.get('final_response'):
                st.markdown("---")
                st.markdown("**Final Response:**")
                st.markdown(str(conv['final_response']))

def display_tool_response(response_data):
    """Display tool response in a structured format"""
    try:
        if not response_data:
            st.text("No response data available")
            return
            
        if isinstance(response_data, str):
            try:
                # Try to parse JSON string
                data = json.loads(response_data)
                st.json(data)
            except:
                # If not JSON, display as text
                st.text(response_data)
        elif isinstance(response_data, dict):
            # Display dictionary content
            for key, value in response_data.items():
                st.markdown(f"**{key}:**")
                if isinstance(value, dict):
                    st.json(value)
                elif isinstance(value, list):
                    for item in value:
                        st.markdown("- " + str(item))
                else:
                    st.markdown(str(value))
        elif isinstance(response_data, list):
            # Display list content
            for item in response_data:
                if isinstance(item, dict):
                    st.json(item)
                else:
                    st.markdown("- " + str(item))
        else:
            # Fallback for other types
            st.text(str(response_data))
    except Exception as e:
        st.error(f"Error displaying response: {str(e)}")

def main():
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📁 Upload & Tools")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Upload file", 
            type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4', 'pdf']
        )
        
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path and st.session_state.agent.set_file_path(file_path):
                st.success(f"Uploaded: {uploaded_file.name}")
                st.session_state.current_file = file_path

        # Display current file if any
        if hasattr(st.session_state, 'current_file'):
            st.info(f"Current file: {os.path.basename(st.session_state.current_file)}")

        # Conversation History
        st.markdown("---")
        conversations = enhanced_logger.get_recent_conversations(limit=5)
        # display_conversation_history(conversations)


        #refrash button 
              # Reset button
        if st.button("Refresh"):
            if hasattr(st.session_state, 'current_file'):
                try:
                    os.unlink(st.session_state.current_file)
                except:
                    pass
                del st.session_state.current_file
            st.session_state.messages = []
            # enhanced_logger.clear_logs()
            st.session_state.agent = AIAgent()
            st.rerun()

        # # Reset button
        # if st.button("Reset to clear logs"):
        #     if hasattr(st.session_state, 'current_file'):
        #         try:
        #             os.unlink(st.session_state.current_file)
        #         except:
        #             pass
        #         del st.session_state.current_file
        #     st.session_state.messages = []
        #     enhanced_logger.clear_logs()
        #     st.session_state.agent = AIAgent()
        #     st.rerun()

    # Chat interface
    st.title("🤖 AI Embededness Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Enter your message"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Format query with file path if exists
                    if hasattr(st.session_state, 'current_file'):
                        full_query = f"file: {st.session_state.current_file} | query: {prompt}"
                    else:
                        full_query = prompt

                    # Get response
                    response = st.session_state.agent.process_query(full_query)
                    
                    # Display response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Use the new display function for the response
                    display_tool_response(response)
                    
                    # Force refresh sidebar
                    st.sidebar.empty()
                    with st.sidebar:
                        st.title("📁 Upload & Tools")
                        if hasattr(st.session_state, 'current_file'):
                            st.info(f"Current file: {os.path.basename(st.session_state.current_file)}")
                        st.markdown("---")
                        conversations = enhanced_logger.get_recent_conversations(limit=5)
                        display_conversation_history(conversations)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()