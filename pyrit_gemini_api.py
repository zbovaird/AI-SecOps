#!/usr/bin/env python3
"""
PyRIT Integration for Gemini 2.5 Flash via Google AI Studio API
Uses API key authentication (not Vertex AI endpoint)
"""

import os
from typing import Optional

class GeminiAPITarget:
    """
    PyRIT-compatible chat target for Gemini 2.5 Flash via Vertex AI
    Uses OAuth2 credentials (not API keys)
    """
    
    def __init__(self, 
                 project_id: Optional[str] = None,
                 region: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini via Vertex AI
        
        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var or ADC)
            region: GCP region (defaults to us-west2)
            model_name: Model name (default: gemini-2.0-flash-exp)
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from google.auth import default
            
            # Get credentials and project
            credentials, default_project = default()
            
            self.project_id = project_id or default_project or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.region = region or os.getenv("GCP_REGION", "us-central1")  # us-central1 has better model availability
            self.model_name = model_name
            
            if not self.project_id:
                raise ValueError(
                    "Project ID required. Set GOOGLE_CLOUD_PROJECT environment variable "
                    "or pass project_id parameter"
                )
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            
            # Load the Gemini model
            self.model = GenerativeModel(model_name)
            
        except ImportError as e:
            raise ImportError(
                "vertexai not installed. Install with: pip install google-cloud-aiplatform"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}") from e
    
    def send_prompt(self, prompt_text: str, **kwargs):
        """
        Send a prompt to Gemini via Vertex AI and get response
        
        Args:
            prompt_text: The prompt to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response text from the model
        """
        try:
            # Prepare generation config
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs or "maxOutputTokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.get("max_tokens") or kwargs.get("maxOutputTokens", 2048)
            if "top_p" in kwargs:
                generation_config["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                generation_config["top_k"] = kwargs["top_k"]
            
            # Generate content using Vertex AI SDK
            response = self.model.generate_content(
                prompt_text,
                generation_config=generation_config if generation_config else None
            )
            
            # Extract text from response
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"Error calling Gemini API: {e}"
    
    def test_connection(self):
        """Test the connection to Gemini API"""
        try:
            test_response = self.send_prompt("Say hello in one word")
            if test_response and not test_response.startswith("Error"):
                return True, test_response
            else:
                return False, test_response
        except Exception as e:
            return False, str(e)



