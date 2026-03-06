"""
MT5 Bridge Server (Windows-side)
================================

TCP server that wraps the MetaTrader5 Python package.
Runs on a Windows machine alongside the MT5 terminal.
Accepts connections from the Traot running on Linux.

Usage:
    python -m src.brokerages.mt5_bridge.server --host 127.0.0.1 --port 5555

The server listens for JSON requests, executes MT5 functions, and returns
JSON responses. It handles:
- initialize / shutdown
- login
- copy_rates_from_pos / copy_rates_range
- symbol_info_tick / symbol_info
- order_send / positions_get / orders_get
- account_info / terminal_info
- symbol_select
- last_error
"""

import argparse
import logging
import socket
import threading
from typing import Any

import numpy as np

from .protocol import MT5Request, MT5Response, read_message

logger = logging.getLogger(__name__)

# Methods that are safe to call
ALLOWED_METHODS = {
    'initialize', 'shutdown', 'login',
    'copy_rates_from_pos', 'copy_rates_range', 'copy_rates_from',
    'copy_ticks_from', 'copy_ticks_range',
    'symbol_info_tick', 'symbol_info', 'symbol_select',
    'symbols_get', 'symbols_total',
    'order_send', 'order_check',
    'positions_get', 'positions_total',
    'orders_get', 'orders_total',
    'history_orders_get', 'history_deals_get',
    'account_info', 'terminal_info',
    'last_error',
}


def _serialize_result(result: Any) -> Any:
    """
    Serialize MT5 result for JSON transport.

    Handles numpy arrays, named tuples, and other MT5-specific types.
    """
    if result is None:
        return None

    # Numpy structured array (from copy_rates_*)
    if isinstance(result, np.ndarray):
        return [
            {name: _to_python(row[name]) for name in result.dtype.names}
            for row in result
        ]

    # Named tuple (from account_info, symbol_info, etc.)
    if hasattr(result, '_asdict'):
        return {k: _to_python(v) for k, v in result._asdict().items()}

    # Tuple of named tuples (from positions_get, orders_get)
    if isinstance(result, tuple):
        return [
            {k: _to_python(v) for k, v in item._asdict().items()}
            if hasattr(item, '_asdict') else _to_python(item)
            for item in result
        ]

    return _to_python(result)


def _to_python(val: Any) -> Any:
    """Convert numpy/MT5 types to native Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val


class MT5BridgeServer:
    """
    TCP server wrapping MetaTrader5 Python package.

    Listens on a configurable port. Receives JSON requests,
    calls MT5 functions, returns JSON responses.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 5555):
        self._host = host
        self._port = port
        self._server_socket: socket.socket = None
        self._running = False
        self._mt5 = None
        self._mt5_lock = threading.Lock()

    def start(self) -> None:
        """Start the bridge server."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "This server must run on Windows with MT5 terminal."
            )

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(2)  # Max 2 pending connections (provider + brokerage)
        self._running = True

        logger.info(f"MT5 Bridge Server listening on {self._host}:{self._port}")
        print(f"MT5 Bridge Server listening on {self._host}:{self._port}")

        try:
            while self._running:
                self._server_socket.settimeout(1.0)
                try:
                    client_sock, addr = self._server_socket.accept()
                    logger.info(f"Client connected from {addr}")
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_sock, addr),
                        daemon=True,
                    )
                    thread.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the bridge server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        if self._mt5:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        logger.info("MT5 Bridge Server stopped")

    def _handle_client(self, client_sock: socket.socket, addr: tuple) -> None:
        """Handle a single client connection."""
        try:
            while self._running:
                msg = read_message(client_sock)
                if msg is None:
                    break  # Client disconnected

                request = MT5Request(
                    id=msg['id'],
                    method=msg['method'],
                    args=msg.get('args', []),
                    kwargs=msg.get('kwargs', {}),
                )

                response = self._process_request(request)
                client_sock.sendall(response.to_bytes())

        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_sock.close()
            logger.info(f"Client {addr} disconnected")

    def _process_request(self, request: MT5Request) -> MT5Response:
        """Process a single MT5 request."""
        if request.method not in ALLOWED_METHODS:
            return MT5Response(
                id=request.id,
                success=False,
                error=f"Method not allowed: {request.method}",
            )

        try:
            with self._mt5_lock:
                func = getattr(self._mt5, request.method)
                result = func(*request.args, **request.kwargs)

            serialized = _serialize_result(result)

            return MT5Response(
                id=request.id,
                success=True,
                data=serialized,
            )

        except Exception as e:
            logger.error(f"Error processing {request.method}: {e}")
            return MT5Response(
                id=request.id,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
            )


def main():
    """Entry point for running the bridge server."""
    parser = argparse.ArgumentParser(description='MT5 Bridge Server')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (use 0.0.0.0 for remote access)')
    parser.add_argument('--port', type=int, default=5555, help='Port number')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    server = MT5BridgeServer(host=args.host, port=args.port)
    server.start()


if __name__ == '__main__':
    main()
