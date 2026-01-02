# Process Resilience in Red Team Kit

## Overview

The Red Team Kit now includes automatic process resilience handling to deal with system resource limits that cause processes to be killed (Killed: 9 errors).

## How It Works

Each module that performs resource-intensive operations now includes:

1. **Primary Execution**: Tries the normal operation first
2. **Retry Logic**: If killed, retries up to 2 times with delays
3. **Python Fallback**: Uses a simpler Python implementation if primary fails
4. **Shell Fallback**: Falls back to shell scripts if Python fails

## Modules Updated

### PostExploitation Module

- `harvest_credentials()` now uses resilience wrapper
- Falls back to simplified credential search if primary fails
- Shell script fallback available for credential harvesting

### ReconModule

- Resilience wrapper added for network operations
- Subprocess operations use resilience handling

## Usage

The resilience is automatic - no code changes needed in your scripts:

```python
from core.modules.post_exploit import PostExploitation
from utils.logger import FrameworkLogger

logger = FrameworkLogger('test')
post_exploit = PostExploitation(logger)

# This will automatically handle process kills
results = post_exploit.harvest_credentials()
```

## Fallback Methods

### Credential Harvesting Fallbacks

1. **Primary**: Full file system search with stealth access
2. **Python Fallback**: Simplified search of accessible paths only
3. **Shell Fallback**: Uses `grep` and shell commands to search

### Network Operations Fallbacks

1. **Primary**: Python socket operations
2. **Shell Fallback**: Uses `netcat`, `timeout`, and bash redirection

## Configuration

You can adjust resilience settings:

```python
from utils.process_resilience import ProcessResilience

resilience = ProcessResilience(logger)
resilience.max_retries = 3  # Default: 2
resilience.retry_delay = 2.0  # Default: 1.0 seconds
```

## Error Handling

When all methods fail, the module returns a structured error:

```python
{
    "status": "failed",
    "error": "Process killed and all fallbacks failed",
    "error_type": "process_killed"
}
```

## Benefits

- **Automatic Recovery**: No manual intervention needed
- **Multiple Fallbacks**: Three layers of fallback protection
- **Shell Script Support**: Can use native shell tools when Python fails
- **Graceful Degradation**: Returns partial results if possible

## Adding Resilience to New Modules

To add resilience to a new module:

```python
from utils.process_resilience import ProcessResilience, ShellBackend

class MyModule:
    def __init__(self, logger):
        self.logger = logger
        self.resilience = ProcessResilience(logger)
    
    def my_method(self):
        return self.resilience.execute_with_resilience(
            self._my_method_primary,
            fallback_func=self._my_method_fallback,
            shell_fallback=ShellBackend.my_shell_fallback()
        )
    
    def _my_method_primary(self):
        # Primary implementation
        pass
    
    def _my_method_fallback(self):
        # Python fallback implementation
        pass
```

## Shell Backend

The `ShellBackend` class provides shell script fallbacks for common operations:

- `harvest_credentials_shell()`: Shell-based credential search
- `network_scan_shell()`: Shell-based port scanning
- `ping_shell()`: Shell-based ping

## Testing

To test resilience:

```python
# This will trigger fallbacks if primary fails
results = module.method()
if results.get("status") == "failed":
    print(f"Error: {results.get('error')}")
else:
    print("Success!")
```

