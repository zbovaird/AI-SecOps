"""
API Clients for Closed-Weight Models

Provides unified interfaces for testing attacks on closed-weight models
via their APIs (OpenAI GPT-4, Anthropic Claude, etc.)
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("transferability.api_clients")


@dataclass
class APIResponse:
    """Standard response from API client."""
    model_id: str
    prompt: str
    response: str
    generation_time_ms: float
    tokens_used: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "prompt": self.prompt[:200],
            "response": self.response[:500],
            "generation_time_ms": self.generation_time_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """Generate a response for a prompt."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[APIResponse]:
        """Generate responses for multiple prompts."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if API connection is working."""
        pass


class OpenAIClient(BaseAPIClient):
    """
    Client for OpenAI API (GPT-4, GPT-4o, etc.)
    
    Usage:
        client = OpenAIClient(api_key="sk-...")
        response = client.generate("Hello, how are you?")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-4o, gpt-4-turbo, etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        self._client = None
        self._initialize_client()
        
        logger.info(f"OpenAI client initialized: model={model}")
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self._client = None
    
    def test_connection(self) -> bool:
        """Test if API connection is working."""
        if self._client is None:
            return False
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> APIResponse:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API
            
        Returns:
            APIResponse
        """
        if self._client is None:
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response="",
                generation_time_ms=0,
                error="OpenAI client not initialized",
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response=response.choices[0].message.content or "",
                generation_time_ms=generation_time_ms,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
            
        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            logger.error(f"OpenAI generation error: {e}")
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response="",
                generation_time_ms=generation_time_ms,
                error=str(e),
            )
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        delay_between: float = 0.5,
        **kwargs,
    ) -> List[APIResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            delay_between: Delay between requests (to avoid rate limits)
            **kwargs: Additional parameters
            
        Returns:
            List of APIResponses
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
            
            if i < len(prompts) - 1:
                time.sleep(delay_between)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(prompts)} prompts")
        
        return responses


class AnthropicClient(BaseAPIClient):
    """
    Client for Anthropic API (Claude 3.5 Sonnet, etc.)
    
    Usage:
        client = AnthropicClient(api_key="sk-ant-...")
        response = client.generate("Hello, how are you?")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        self._client = None
        self._initialize_client()
        
        logger.info(f"Anthropic client initialized: model={model}")
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            logger.warning("anthropic package not installed. Install with: pip install anthropic")
            self._client = None
    
    def test_connection(self) -> bool:
        """Test if API connection is working."""
        if self._client is None:
            return False
        
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> APIResponse:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            APIResponse
        """
        if self._client is None:
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response="",
                generation_time_ms=0,
                error="Anthropic client not initialized",
            )
        
        start_time = time.time()
        
        try:
            create_kwargs = {
                "model": kwargs.get("model", self.model),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}],
            }
            
            if system_prompt:
                create_kwargs["system"] = system_prompt
            
            # Only add temperature if not default
            temp = kwargs.get("temperature", self.temperature)
            if temp != 1.0:
                create_kwargs["temperature"] = temp
            
            response = self._client.messages.create(**create_kwargs)
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Extract text from response
            response_text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_text += block.text
            
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response=response_text,
                generation_time_ms=generation_time_ms,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens if response.usage else 0,
            )
            
        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Anthropic generation error: {e}")
            return APIResponse(
                model_id=self.model,
                prompt=prompt,
                response="",
                generation_time_ms=generation_time_ms,
                error=str(e),
            )
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        delay_between: float = 0.5,
        **kwargs,
    ) -> List[APIResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            delay_between: Delay between requests
            **kwargs: Additional parameters
            
        Returns:
            List of APIResponses
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
            
            if i < len(prompts) - 1:
                time.sleep(delay_between)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(prompts)} prompts")
        
        return responses


class UnifiedAPIClient:
    """
    Unified interface for testing across multiple API providers.
    
    Usage:
        client = UnifiedAPIClient(
            openai_key="sk-...",
            anthropic_key="sk-ant-...",
        )
        results = client.test_all(prompts)
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        anthropic_model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize unified client.
        
        Args:
            openai_key: OpenAI API key (optional)
            anthropic_key: Anthropic API key (optional)
            openai_model: OpenAI model to use
            anthropic_model: Anthropic model to use
        """
        self.clients: Dict[str, BaseAPIClient] = {}
        
        if openai_key:
            self.clients["openai"] = OpenAIClient(openai_key, model=openai_model)
        
        if anthropic_key:
            self.clients["anthropic"] = AnthropicClient(anthropic_key, model=anthropic_model)
        
        logger.info(f"UnifiedAPIClient initialized with {len(self.clients)} providers")
    
    def test_connections(self) -> Dict[str, bool]:
        """Test all API connections."""
        return {name: client.test_connection() for name, client in self.clients.items()}
    
    def generate_all(
        self,
        prompt: str,
        **kwargs,
    ) -> Dict[str, APIResponse]:
        """
        Generate responses from all configured providers.
        
        Args:
            prompt: The prompt to test
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping provider name to APIResponse
        """
        return {
            name: client.generate(prompt, **kwargs)
            for name, client in self.clients.items()
        }
    
    def generate_batch_all(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, List[APIResponse]]:
        """
        Generate batch responses from all providers.
        
        Args:
            prompts: List of prompts
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping provider name to list of APIResponses
        """
        results = {}
        
        for name, client in self.clients.items():
            logger.info(f"Processing {name}...")
            results[name] = client.generate_batch(prompts, **kwargs)
        
        return results
