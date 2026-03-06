"""
Order Events (Lean-Inspired)
============================
Event system for order status notifications.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orders import Order


class OrderEventType(Enum):
    """Order event types (Lean-compatible)."""
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partial_fill"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UPDATED = "updated"


@dataclass
class OrderEvent:
    """
    Order event notification (Lean-inspired).

    Fired when order status changes. Register callbacks
    to receive these events.

    Example:
        def on_order(event: OrderEvent):
            if event.event_type == OrderEventType.FILLED:
                print(f"Filled {event.fill_quantity} @ {event.fill_price}")

        brokerage.on_order_event(on_order)
    """
    order_id: str
    event_type: OrderEventType
    order: 'Order'

    # Fill info (for fill events)
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    commission: float = 0.0

    # Metadata
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_fill(self) -> bool:
        """Check if this is a fill event."""
        return self.event_type in [
            OrderEventType.FILLED,
            OrderEventType.PARTIALLY_FILLED
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal event (order complete)."""
        return self.event_type in [
            OrderEventType.FILLED,
            OrderEventType.CANCELED,
            OrderEventType.REJECTED,
            OrderEventType.EXPIRED
        ]

    def __repr__(self) -> str:
        return (
            f"OrderEvent({self.order_id}: {self.event_type.value} "
            f"@ {self.timestamp.strftime('%H:%M:%S')})"
        )
