#!/usr/bin/env python3
"""
PyRIT Vertex AI Integration
Provides a Vertex AI endpoint interface that PyRIT can use
"""

import os
from typing import Optional

class VertexAIChatTarget:
    """
    A PyRIT-compatible chat target for Vertex AI
    """
    
    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None, model_name: str = "text-bison@001"):
        """
        Initialize Vertex AI chat target
        
        Args:
            project_id: GCP project ID (defaults to environment or ADC)
            region: GCP region (defaults to us-central1)
            model_name: Vertex AI model name
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from google.auth import default
            
            # Get credentials and project
            credentials, default_project = default()
            
            self.project_id = project_id or default_project or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.region = region or os.getenv("GCP_REGION", "us-central1")
            self.model_name = model_name
            
            if not self.project_id:
                raise ValueError("Project ID required. Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id")
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            
            # Load the model - using Gemini for text generation
            # For text-bison, we'll use the GenerativeModel API
            if "gemini" in model_name.lower():
                self.model = GenerativeModel(model_name)
            else:
                # Fallback: use gemini-pro as default
                self.model = GenerativeModel("gemini-pro")
            
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform not installed. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vertex AI: {e}") from e
    
    def send_prompt(self, prompt_text: str, **kwargs):
        """
        Send a prompt to Vertex AI and get response
        
        Args:
            prompt_text: The prompt to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response text from the model
        """
        try:
            # Set default parameters
            parameters = {
                "temperature": kwargs.get("temperature", 0.7),
                "max_output_tokens": kwargs.get("max_output_tokens", 1024),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
            }
            
            # Update with any provided kwargs
            parameters.update({k: v for k, v in kwargs.items() 
                             if k in ["temperature", "max_output_tokens", "top_p", "top_k"]})
            
            # Get response from model
            if isinstance(self.model, GenerativeModel):
                # Gemini API
                response = self.model.generate_content(prompt_text, generation_config=parameters)
                return response.text if hasattr(response, 'text') else str(response)
            else:
                # Fallback for other model types
                response = self.model.predict(prompt_text, **parameters)
                return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"Error calling Vertex AI: {e}"
    
    def test_connection(self):
        """Test the connection to Vertex AI"""
        try:
            test_response = self.send_prompt("Say hello")
            return True, test_response
        except Exception as e:
            return False, str(e)



