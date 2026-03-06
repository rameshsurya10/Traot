"""
Resilience and Error Recovery (Production-Grade)
================================================
Error handling, retry logic, and automatic recovery for live trading.

Features:
- Circuit breaker pattern
- Exponential backoff retry
- Health monitoring
- Automatic reconnection
- Graceful degradation

Inspired by production trading systems and reliability patterns.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 2      # Successes to close from half-open
    timeout_seconds: float = 60.0   # Time before trying half-open
    excluded_exceptions: List[type] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit Breaker Pattern.

    Prevents cascading failures by temporarily blocking operations
    when too many failures occur.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, requests rejected immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker("broker_api")

        @breaker
        def call_broker():
            return api.get_positions()

        # Or manually
        if breaker.can_execute():
            try:
                result = call_broker()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure(e)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Monitoring
        self._total_calls = 0
        self._total_failures = 0

        logger.info(f"CircuitBreaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal)."""
        return self.state == CircuitState.CLOSED

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout elapsed
                if self._last_failure_time:
                    elapsed = datetime.utcnow() - self._last_failure_time
                    if elapsed.total_seconds() >= self.config.timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        logger.info(f"Circuit '{self.name}' entering HALF_OPEN")
                        return True
                return False

            # HALF_OPEN: allow one request through
            return True

    def record_success(self):
        """Record successful execution."""
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset failures on success

    def record_failure(self, exception: Exception = None):
        """Record failed execution."""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1

            # Check if exception is excluded
            if exception and any(
                isinstance(exception, exc) for exc in self.config.excluded_exceptions
            ):
                return

            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                self._open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._open()

    def _open(self):
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._success_count = 0
        logger.warning(f"Circuit '{self.name}' OPENED after {self._failure_count} failures")

    def _close(self):
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit '{self.name}' CLOSED - service recovered")

    def reset(self):
        """Manually reset circuit."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise CircuitBreakerOpen(f"Circuit '{self.name}' is OPEN")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'total_calls': self._total_calls,
                'total_failures': self._total_failures,
                'failure_rate': (
                    f"{(self._total_failures / self._total_calls * 100):.1f}%"
                    if self._total_calls > 0 else "0%"
                ),
            }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0        # Initial delay in seconds
    max_delay: float = 60.0        # Maximum delay
    exponential_base: float = 2.0  # Backoff multiplier
    jitter: float = 0.1            # Random jitter (0-1)
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])


def retry_with_backoff(
    config: RetryConfig = None,
    on_retry: Callable[[int, Exception], None] = None
):
    """
    Retry decorator with exponential backoff.

    Example:
        @retry_with_backoff(RetryConfig(max_retries=5))
        def unreliable_operation():
            return api.call()
    """
    config = config or RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except tuple(config.retryable_exceptions) as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    # Add jitter
                    import random
                    jitter = delay * config.jitter * random.random()
                    delay += jitter

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.1f}s: {e}"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class HealthMonitor:
    """
    Health Monitor for Trading System.

    Tracks component health and triggers alerts.
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3
    ):
        """
        Args:
            check_interval: Seconds between health checks
            unhealthy_threshold: Consecutive failures before unhealthy
        """
        self._components: Dict[str, dict] = {}
        self._check_interval = check_interval
        self._unhealthy_threshold = unhealthy_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_unhealthy: List[Callable] = []
        self._on_recovered: List[Callable] = []

    def register_component(
        self,
        name: str,
        check_func: Callable[[], bool],
        critical: bool = True
    ):
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            check_func: Function returning True if healthy
            critical: Whether component is critical
        """
        with self._lock:
            self._components[name] = {
                'check_func': check_func,
                'critical': critical,
                'healthy': True,
                'failure_count': 0,
                'last_check': None,
                'last_error': None,
            }

    def unregister_component(self, name: str):
        """Unregister a component."""
        with self._lock:
            if name in self._components:
                del self._components[name]

    def check_health(self, name: str) -> bool:
        """Check health of specific component."""
        with self._lock:
            if name not in self._components:
                return False

            comp = self._components[name]

        try:
            healthy = comp['check_func']()
            comp['last_check'] = datetime.utcnow()

            if healthy:
                if not comp['healthy'] and comp['failure_count'] > 0:
                    # Component recovered
                    self._notify_recovered(name)
                comp['healthy'] = True
                comp['failure_count'] = 0
                comp['last_error'] = None
            else:
                comp['failure_count'] += 1
                if comp['failure_count'] >= self._unhealthy_threshold:
                    if comp['healthy']:
                        self._notify_unhealthy(name, "Health check returned False")
                    comp['healthy'] = False

            return comp['healthy']

        except Exception as e:
            comp['failure_count'] += 1
            comp['last_error'] = str(e)
            comp['last_check'] = datetime.utcnow()

            if comp['failure_count'] >= self._unhealthy_threshold:
                if comp['healthy']:
                    self._notify_unhealthy(name, str(e))
                comp['healthy'] = False

            return False

    def check_all(self) -> Dict[str, bool]:
        """Check health of all components."""
        results = {}
        for name in list(self._components.keys()):
            results[name] = self.check_health(name)
        return results

    @property
    def is_healthy(self) -> bool:
        """Check if all critical components are healthy."""
        with self._lock:
            for name, comp in self._components.items():
                if comp['critical'] and not comp['healthy']:
                    return False
        return True

    def on_unhealthy(self, callback: Callable[[str, str], None]):
        """Register unhealthy callback (component_name, error)."""
        self._on_unhealthy.append(callback)

    def on_recovered(self, callback: Callable[[str], None]):
        """Register recovery callback (component_name)."""
        self._on_recovered.append(callback)

    def _notify_unhealthy(self, name: str, error: str):
        """Notify unhealthy callbacks."""
        logger.error(f"Component '{name}' is UNHEALTHY: {error}")
        for callback in self._on_unhealthy:
            try:
                callback(name, error)
            except Exception as e:
                logger.error(f"Unhealthy callback error: {e}")

    def _notify_recovered(self, name: str):
        """Notify recovery callbacks."""
        logger.info(f"Component '{name}' RECOVERED")
        for callback in self._on_recovered:
            try:
                callback(name)
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")

    def start(self):
        """Start background health checking."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._thread.start()
        logger.info("HealthMonitor started")

    def stop(self):
        """Stop health checking."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            time.sleep(self._check_interval)

    def get_status(self) -> dict:
        """Get health status of all components."""
        with self._lock:
            return {
                'healthy': self.is_healthy,
                'components': {
                    name: {
                        'healthy': comp['healthy'],
                        'critical': comp['critical'],
                        'failure_count': comp['failure_count'],
                        'last_check': comp['last_check'].isoformat() if comp['last_check'] else None,
                        'last_error': comp['last_error'],
                    }
                    for name, comp in self._components.items()
                }
            }


class ReconnectionManager:
    """
    Automatic Reconnection Manager.

    Handles reconnection to brokerages and data feeds.
    """

    def __init__(
        self,
        max_attempts: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,  # 5 minutes
        backoff_factor: float = 2.0
    ):
        """
        Args:
            max_attempts: Maximum reconnection attempts (0 = unlimited)
            initial_delay: Initial delay between attempts
            max_delay: Maximum delay between attempts
            backoff_factor: Delay multiplier
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        self._connections: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def register_connection(
        self,
        name: str,
        connect_func: Callable[[], bool],
        disconnect_func: Callable[[], None] = None,
        on_connected: Callable[[], None] = None,
        on_failed: Callable[[str], None] = None
    ):
        """
        Register a connection for automatic reconnection.

        Args:
            name: Connection name
            connect_func: Function to establish connection, returns True if successful
            disconnect_func: Function to cleanly disconnect
            on_connected: Callback when connection established
            on_failed: Callback when all reconnection attempts fail
        """
        with self._lock:
            self._connections[name] = {
                'connect_func': connect_func,
                'disconnect_func': disconnect_func,
                'on_connected': on_connected,
                'on_failed': on_failed,
                'connected': False,
                'reconnecting': False,  # Prevent concurrent reconnection
                'attempts': 0,
                'last_attempt': None,
            }

    def reconnect(self, name: str) -> bool:
        """
        Attempt reconnection with exponential backoff.

        Args:
            name: Connection name

        Returns:
            True if reconnection successful
        """
        # Thread-safe check and acquire reconnection lock
        with self._lock:
            if name not in self._connections:
                return False
            conn = self._connections[name]

            # Prevent concurrent reconnection attempts
            if conn.get('reconnecting', False):
                logger.warning(f"Reconnection already in progress for {name}")
                return False
            conn['reconnecting'] = True

        try:
            # Disconnect first if connected
            if conn['connected'] and conn['disconnect_func']:
                try:
                    conn['disconnect_func']()
                except Exception as e:
                    logger.warning(f"Disconnect error for {name}: {e}")

            with self._lock:
                conn['connected'] = False
                conn['attempts'] = 0

            # Track start time for max duration protection
            start_time = datetime.utcnow()
            max_duration = timedelta(hours=1)  # Maximum 1 hour of retrying

            while True:
                with self._lock:
                    conn['attempts'] += 1
                    conn['last_attempt'] = datetime.utcnow()
                    current_attempts = conn['attempts']

                # Check max duration (protects against infinite loop)
                if datetime.utcnow() - start_time > max_duration:
                    logger.error(f"Max reconnection duration exceeded for {name}")
                    if conn['on_failed']:
                        conn['on_failed']("Max duration (1 hour) exceeded")
                    return False

                # Check max attempts
                if self.max_attempts > 0 and current_attempts > self.max_attempts:
                    logger.error(f"Max reconnection attempts reached for {name}")
                    if conn['on_failed']:
                        conn['on_failed'](f"Max attempts ({self.max_attempts}) reached")
                    return False

                # Calculate delay
                if current_attempts > 1:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** (current_attempts - 1)),
                        self.max_delay
                    )
                    logger.info(f"Reconnecting {name} in {delay:.1f}s (attempt {current_attempts})")
                    time.sleep(delay)

                try:
                    logger.info(f"Attempting reconnection for {name}")
                    if conn['connect_func']():
                        with self._lock:
                            conn['connected'] = True
                            conn['attempts'] = 0
                        logger.info(f"Reconnected to {name}")
                        if conn['on_connected']:
                            conn['on_connected']()
                        return True

                except Exception as e:
                    logger.error(f"Reconnection attempt {current_attempts} failed for {name}: {e}")

            return False

        finally:
            # Always release reconnection lock
            with self._lock:
                conn['reconnecting'] = False

    def reconnect_async(self, name: str) -> threading.Thread:
        """Start reconnection in background thread."""
        thread = threading.Thread(
            target=self.reconnect,
            args=(name,),
            daemon=True,
            name=f"Reconnect-{name}"
        )
        thread.start()
        return thread

    def is_connected(self, name: str) -> bool:
        """Check if connection is active."""
        with self._lock:
            if name in self._connections:
                return self._connections[name]['connected']
        return False

    def get_status(self) -> dict:
        """Get status of all connections."""
        with self._lock:
            return {
                name: {
                    'connected': conn['connected'],
                    'attempts': conn['attempts'],
                    'last_attempt': conn['last_attempt'].isoformat() if conn['last_attempt'] else None,
                }
                for name, conn in self._connections.items()
            }


class RateLimiter:
    """
    Rate Limiter for API calls.

    Prevents exceeding API rate limits.
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        name: str = "default"
    ):
        """
        Args:
            max_calls: Maximum calls allowed in period
            period_seconds: Time period in seconds
            name: Limiter name for logging
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.name = name

        self._calls: deque = deque()
        self._lock = threading.Lock()

    def can_call(self) -> bool:
        """Check if a call is allowed."""
        with self._lock:
            self._cleanup()
            return len(self._calls) < self.max_calls

    def record_call(self):
        """Record a call."""
        with self._lock:
            self._calls.append(time.time())

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit exceeded.

        Returns:
            Seconds waited (0 if no wait needed)
        """
        with self._lock:
            self._cleanup()

            if len(self._calls) < self.max_calls:
                return 0.0

            # Calculate wait time
            oldest = self._calls[0]
            wait_time = (oldest + self.period_seconds) - time.time()

            if wait_time > 0:
                logger.debug(f"Rate limit '{self.name}': waiting {wait_time:.2f}s")

        if wait_time > 0:
            time.sleep(wait_time)
            return wait_time

        return 0.0

    def _cleanup(self):
        """Remove old calls from tracking."""
        cutoff = time.time() - self.period_seconds
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            self.record_call()
            return func(*args, **kwargs)
        return wrapper

    def get_status(self) -> dict:
        """Get rate limiter status."""
        with self._lock:
            self._cleanup()
            return {
                'name': self.name,
                'calls_in_window': len(self._calls),
                'max_calls': self.max_calls,
                'period_seconds': self.period_seconds,
                'utilization': f"{len(self._calls) / self.max_calls * 100:.1f}%",
            }
