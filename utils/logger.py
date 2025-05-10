# logger.py
import logging
import os
from datetime import datetime
import json
import csv
import pandas as pd
from typing import Dict, Any

class EnhancedLogger:
    def __init__(self):
        self.log_dir = 'logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Setup logging files with timestamps to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.log_filename = f'agent_{timestamp}.log'
        # self.csv_filename = f'conversation_{timestamp}.csv'
        self.log_filename = f'agent.log'
        self.csv_filename = f'conversation.csv'
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        self.csv_path = os.path.join(self.log_dir, self.csv_filename)
        
        # Configure standard logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ai_agent')
        
        # Initialize CSV if it doesn't exist
        self._initialize_csv()
        self.logger.info(f"Logger initialized. CSV path: {self.csv_path}")

    def _initialize_csv(self):
        """Initialize CSV with headers"""
        try:
            headers = [
                'timestamp',
                'conversation_id',
                'user_query',
                'file_path',
                'file_type',
                'tool_name',
                'tool_arguments',
                'tool_response',
                'final_response'
            ]
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.logger.info("CSV file initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing CSV: {str(e)}")

    def log_conversation(self, 
                        user_query: str,
                        file_path: str = None,
                        tool_name: str = None,
                        tool_args: Dict = None,
                        tool_response: Any = None,
                        final_response: str = None,
                        conversation_id: str = None):
        """Log a complete conversation entry to CSV"""
        try:
            timestamp = datetime.now().isoformat()
            file_type = os.path.splitext(file_path)[1] if file_path else None
            
            # Prepare row data
            row_data = {
                'timestamp': timestamp,
                'conversation_id': conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                'user_query': user_query,
                'file_path': file_path,
                'file_type': file_type,
                'tool_name': tool_name,
                'tool_arguments': json.dumps(tool_args) if tool_args else None,
                'tool_response': json.dumps(tool_response) if tool_response else None,
                'final_response': final_response
            }
            
            # Write to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                writer.writerow(row_data)
            
            # Log success
            self.logger.info(f"Successfully logged conversation: {json.dumps(row_data)}")
            return row_data
            
        except Exception as e:
            self.logger.error(f"Error logging conversation: {str(e)}")
            return None

    def get_recent_conversations(self, limit: int = 5) -> list:
        """Get recent conversations from CSV"""
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path, encoding='utf-8')
                if df.empty:
                    self.logger.info("No conversations found in CSV")
                    return []
                conversations = df.tail(limit).to_dict('records')
                self.logger.info(f"Retrieved {len(conversations)} recent conversations")
                return conversations
        except Exception as e:
            self.logger.error(f"Error reading conversations: {str(e)}")
        return []

    def clear_logs(self):
        """Clear both CSV and log file"""
        try:
            # Clear log file
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write('')
            
            # Reinitialize CSV
            self._initialize_csv()
            
            self.logger.info("Logs cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing logs: {str(e)}")

# Create singleton instance
enhanced_logger = EnhancedLogger()