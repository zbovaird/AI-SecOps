"""
Advanced Payloads Module
Generates advanced payloads for security testing

Usage for Red Teaming:
---------------------
The Advanced Payloads module generates encoded and obfuscated payloads to evade
detection systems. It supports Base64, ROT13, hexadecimal, multi-layer encoding,
and noise injection. This helps test payload detection systems and evasion
capabilities.

Example Usage:
    from core.modules.advanced_payloads import AdvancedPayloads

    # Initialize module
    payloads = AdvancedPayloads()

    # Generate encoded payload
    command = "execute malicious code"
    base64_payload = payloads.generate_encoded_payload(command)
    print(f"Base64: {base64_payload}")

    # Generate ROT13 payload
    rot13_payload = payloads.generate_rot13_payload(command)
    print(f"ROT13: {rot13_payload}")

    # Generate hex payload
    hex_payload = payloads.generate_hex_payload(command)
    print(f"Hex: {hex_payload}")

    # Generate multi-layer encoded payload
    multilayer = payloads.generate_multilayer_payload(command)
    print(f"Multi-layer: {multilayer}")

    # Generate variants
    variants = payloads.generate_variants(command, num_variants=5)
    print(f"Generated {len(variants)} payload variants")

    # View payload history
    history = payloads.get_payload_history()
    print(f"Payload history: {len(history)} entries")

Red Team Use Cases:
- Generating encoded payloads to evade detection
- Testing payload detection systems
- Base64 encoding evasion
- ROT13 encoding evasion
- Hexadecimal encoding evasion
- Multi-layer encoding for advanced evasion
- Noise injection to bypass filters
- Payload history tracking for analysis
"""

from typing import List, Dict, Any, Optional
import base64
import random
import string


class AdvancedPayloads:
    """Advanced payload generation"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced payloads module"""
        self.config = config or {}
        self.payload_history = []

    def generate_encoded_payload(self, command: str) -> str:
        """
        Generate base64 encoded payload

        Args:
            command: Command to encode

        Returns:
            Base64 encoded payload string
        """
        encoded = base64.b64encode(command.encode()).decode()
        return f"Execute: {encoded}"

    def generate_obfuscated_payload(self, command: str) -> str:
        """
        Generate obfuscated payload

        Args:
            command: Command to obfuscate

        Returns:
            Obfuscated payload string
        """
        # Simple obfuscation example
        obfuscated = ''.join([chr(ord(c) + 1) if c.isalpha() else c for c in command])
        return f"Run: {obfuscated}"

    def generate_rot13_payload(self, command: str) -> str:
        """
        Generate ROT13 encoded payload

        Args:
            command: Command to encode

        Returns:
            ROT13 encoded payload
        """
        def rot13(text: str) -> str:
            result = []
            for char in text:
                if char.isalpha():
                    ascii_offset = 65 if char.isupper() else 97
                    rotated = (ord(char) - ascii_offset + 13) % 26 + ascii_offset
                    result.append(chr(rotated))
                else:
                    result.append(char)
            return ''.join(result)

        encoded = rot13(command)
        return f"Decode ROT13: {encoded}"

    def generate_hex_payload(self, command: str) -> str:
        """
        Generate hexadecimal encoded payload

        Args:
            command: Command to encode

        Returns:
            Hex encoded payload
        """
        hex_encoded = command.encode().hex()
        return f"Execute hex: {hex_encoded}"

    def generate_multilayer_payload(self, command: str) -> str:
        """
        Generate multi-layer encoded payload

        Args:
            command: Command to encode

        Returns:
            Multi-layer encoded payload
        """
        # Base64 -> Hex -> ROT13
        layer1 = base64.b64encode(command.encode()).decode()
        layer2 = layer1.encode().hex()

        def rot13(text: str) -> str:
            result = []
            for char in text:
                if char.isalpha():
                    ascii_offset = 65 if char.isupper() else 97
                    rotated = (ord(char) - ascii_offset + 13) % 26 + ascii_offset
                    result.append(chr(rotated))
                else:
                    result.append(char)
            return ''.join(result)

        layer3 = rot13(layer2)
        return f"Multi-layer decode: {layer3}"

    def generate_noise_payload(self, command: str, noise_level: int = 5) -> str:
        """
        Generate payload with noise to evade detection

        Args:
            command: Command to hide
            noise_level: Amount of noise to add

        Returns:
            Payload with noise
        """
        noise_chars = random.choices(string.ascii_letters + string.digits, k=noise_level)
        noise = ''.join(noise_chars)
        return f"{noise} {command} {noise}"

    def generate_variants(self, command: str, num_variants: int = 5) -> List[str]:
        """
        Generate multiple payload variants

        Args:
            command: Command to generate variants for
            num_variants: Number of variants to generate

        Returns:
            List of payload variants
        """
        generators = [
            self.generate_encoded_payload,
            self.generate_obfuscated_payload,
            self.generate_rot13_payload,
            self.generate_hex_payload,
            self.generate_noise_payload
        ]

        variants = []
        selected = random.sample(generators, min(num_variants, len(generators)))

        for generator in selected:
            try:
                payload = generator(command)
                variants.append(payload)
                self.payload_history.append({
                    "command": command,
                    "payload": payload,
                    "method": generator.__name__
                })
            except Exception as e:
                continue

        return variants

    def get_payload_history(self) -> List[Dict]:
        """Get payload generation history"""
        return self.payload_history

