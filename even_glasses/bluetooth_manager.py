import asyncio
import logging
import time
from dataclasses import dataclass
from logging import Logger

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError
from typing import Optional, Callable, Coroutine
from even_glasses.models import DesiredConnectionState

from even_glasses.utils import construct_heartbeat
from even_glasses.service_identifiers import (
    UART_SERVICE_UUID,
    UART_TX_CHAR_UUID,
    UART_RX_CHAR_UUID,
)

logging.basicConfig(level=logging.INFO)
_default_logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DiscoveryResult:
    left_glass_name: str
    right_glass_name: str
    left_glass_address: str
    right_glass_address: str

class BleDevice:
    """Base class for BLE device communication."""

    def __init__(self, name: str, address: str, logger: Logger = _default_logger):
        self._logger = logger
        self.name = name
        self.address = address
        self.client = BleakClient(
            address,
            disconnected_callback=self._handle_disconnection,
        )
        self.uart_tx = None
        self.uart_rx = None
        self._write_lock = asyncio.Lock()
        self.notifications_started = False
        self.desired_connection_state = DesiredConnectionState.DISCONNECTED

    async def connect(self):
        self._logger.info(f"Connecting to {self.name} ({self.address})")
        try:
            await self.client.connect()
            self._logger.info(f"Connected to {self.name}")

            # Discover services
            services = self.client.services

            uart_service = services.get_service(UART_SERVICE_UUID)
            if not uart_service:
                raise BleakError(f"UART service not found for {self.name}")

            self.uart_tx = uart_service.get_characteristic(UART_TX_CHAR_UUID)
            self.uart_rx = uart_service.get_characteristic(UART_RX_CHAR_UUID)

            if not self.uart_tx or not self.uart_rx:
                raise BleakError(f"UART TX/RX characteristics not found for {self.name}")

            await self.start_notifications()
        except Exception as e:
            self._logger.error(f"Error connecting to {self.name}: {e}")
            await self.disconnect()
            raise

    async def disconnect(self, raise_exceptions: bool = False):
        """Gracefully disconnect from the BLE device, stopping notifications if they are active."""

        exceptions = []
        try:
            # Check if notifications are started and `uart_rx` exists before stopping them
            if self.notifications_started and self.uart_rx:
                try:
                    await self.client.stop_notify(self.uart_rx)
                    self._logger.info(f"Stopped notifications for {self.name}")
                except Exception as e:
                    self._logger.warning(f"Failed to stop notifications for {self.name}: {e}")
                    exceptions.append(e)
                finally:
                    self.notifications_started = False

            # Check if the client is still connected before attempting to disconnect
            if self.client.is_connected:
                await self.client.disconnect()
                self._logger.info(f"Disconnected from {self.name}")
        except Exception as e:
            self._logger.error(f"Error during disconnection for {self.name}: {e}")
            exceptions.append(e)
        if len(exceptions) == 1:
            raise exceptions[0]
        elif len(exceptions) > 1:
            raise ExceptionGroup("During BleDevice disconnect exceptions occurred.", exceptions)

    def _handle_disconnection(self, client: BleakClient):
        self._logger.warning(f"Device {self.name} disconnected")
        if self.desired_connection_state == DesiredConnectionState.CONNECTED:
            asyncio.create_task(self.reconnect())

    async def reconnect(self):
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                self._logger.info(f"Reconnecting to {self.name} (Attempt {attempt}/{retries})")
                await self.connect()
                self._logger.info(f"Reconnected to {self.name}")
                return
            except Exception as e:
                self._logger.error(f"Reconnection attempt {attempt} failed: {e}")
                await asyncio.sleep(5)
        self._logger.error(f"Failed to reconnect to {self.name} after {retries} attempts")

    async def start_notifications(self):
        if not self.notifications_started and self.uart_rx:
            try:
                await self.client.start_notify(self.uart_rx, self.handle_notification)
                self.notifications_started = True
                self._logger.info(f"Notifications started for {self.name}")
            except Exception as e:
                self._logger.error(f"Failed to start notifications for {self.name}: {e}")

    async def send(self, data: bytes) -> bool:
        start = time.perf_counter()
        if not self.client.is_connected:
            self._logger.warning(f"Cannot send data, {self.name} is disconnected.")
            return False

        if not self.uart_tx:
            self._logger.warning(f"No TX characteristic available for {self.name}.")
            return False
        if len(data) > 253:
            raise ValueError("Message too long, payload must be at most 253 bytes long.")

        try:
            async with self._write_lock:
                await self.client.write_gatt_char(self.uart_tx, data, response=True)
            self._logger.info(f"Data sent to {self.name}, took {time.perf_counter() - start}[s], data[{len(data)}]: {data.hex()}")
            return True
        except Exception as e:
            self._logger.error(f"Error sending data to {self.name}: {e}")
            return False

    async def handle_notification(self, sender: int, data: bytes):
        ...


class Glass(BleDevice):
    """Class representing a single glass device."""

    def __init__(
        self,
        name: str,
        address: str,
        side: str,
        heartbeat_freq: int = 5,
    ):
        super().__init__(name, address)
        self.side = side
        self.heartbeat_freq = heartbeat_freq
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.notification_handler: Optional[Callable[['Glass', int, bytes], Coroutine]] = None
        

    async def start_heartbeat(self):
        if self.heartbeat_task is None or self.heartbeat_task.done():
            self.heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self):
        while self.client.is_connected:
            try:
                heartbeat = construct_heartbeat(1)
                await self.send(heartbeat)
                await asyncio.sleep(self.heartbeat_freq)
            except Exception as e:
                self._logger.error(f"Heartbeat error for {self.name}: {e}")
                break

    async def connect(self):
        await super().connect()
        await self.start_heartbeat()

    async def disconnect(self, throw_exceptions: bool = False):
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        await super().disconnect(throw_exceptions)
    
    async def handle_notification(self, sender: int, data: bytes):
        self._logger.info(f"Notification from {self.name}: {data.hex()}")
        if self.notification_handler:
            await self.notification_handler(self,sender, data)


class GlassesManager:
    """Class to manage both left and right glasses."""
    # TODO: Maybe refactor this class into something that locks glasses after connection
    # TODO: and block changing internal state by external actors util internally managed resources are closed (connections are disconnected)

    def __init__(
        self,
        left_address: str = None,
        right_address: str = None,
        left_name: str = "G1 Left Glass",
        right_name: str = "G1 Right Glass",
        logger: Logger = _default_logger
    ):
        self._logger = logger
        self.left_glass: Optional[Glass] = (
            Glass(name=left_name, address=left_address, side="left")
            if left_address
            else None
        )
        self.right_glass: Optional[Glass] = (
            Glass(name=right_name, address=right_address, side="right")
            if right_address
            else None
        )
        self.desired_connection_state = DesiredConnectionState.DISCONNECTED

    async def connect(self):
        if not self.left_glass:
            raise RuntimeError("Left glass is not configured. Reconstruct the object with predefined connection or call 'scan_and_connect'.")
        if not self.right_glass:
            raise RuntimeError("Right glass is not configured. Reconstruct the object with predefined connection or call 'scan_and_connect'.")
        await self.left_glass.connect()
        await self.right_glass.connect()
        self.desired_connection_state = DesiredConnectionState.CONNECTED
        return True

    @classmethod
    async def scan_for_glasses(cls, *, timeout: float = 10, logger: Logger = _default_logger) -> DiscoveryResult:
        """
        Scan for glasses.
        Throws exception if one of glasses or both are not found.
        """
        devices = await BleakScanner.discover(timeout=timeout)
        left_glass, right_glass, left_glass_address, right_glass_address = None, None, None, None
        for device in devices:
            device_name = device.name or "Unknown"
            logger.info(f"Found device: {device_name}, Address: {device.address}")
            if "_L_" in device_name and not left_glass:
                left_glass = device.name
                left_glass_address = device.address
            elif "_R_" in device_name and not right_glass:
                right_glass = device.name
                right_glass_address = device.address
        if not (left_glass and right_glass):
            raise RuntimeError(f"Failed to find both glasses, left glass: {left_glass}, right_glass: {right_glass}")
        return DiscoveryResult(left_glass_name=left_glass, right_glass_name=right_glass,
                               left_glass_address=left_glass_address, right_glass_address=right_glass_address)

    async def scan_and_connect(self, timeout: float = 10) -> bool:
        """Scan for glasses devices and connect to them."""
        self._logger.info("Scanning for glasses devices...")
        result =  await self.scan_for_glasses(timeout=timeout, logger=self._logger)
        self.left_glass = Glass(result.left_glass_name, result.left_glass_address, 'left')
        self.right_glass = Glass(result.right_glass_name, result.right_glass_address, 'right')
        await self.connect()
        return True

    async def disconnect_all(self, throw_exceptions: bool = False):
        """
        Disconnect from all connected glasses.
        """
        self.desired_connection_state = DesiredConnectionState.DISCONNECTED
        exceptions = []
        if self.left_glass and self.left_glass.client.is_connected:
            try:
                await self.left_glass.disconnect(throw_exceptions)
            except Exception as e:
                exceptions.append(e)
            self._logger.info("Disconnected left glass.")
        if self.right_glass and self.right_glass.client.is_connected:
            try:
                await self.right_glass.disconnect(throw_exceptions)
            except Exception as e:
                exceptions.append(e)
            self._logger.info("Disconnected right glass.")
        if throw_exceptions and len(exceptions) > 0:
            raise ExceptionGroup("When disconnecting glasses errors occurred.", exceptions)

# Example Usage
async def main():
    manager = GlassesManager()
    connected = await manager.scan_and_connect()
    if connected:
        try:
            while True:
                # Replace with your actual logic
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            _default_logger.info("Interrupted by user.")
        finally:
            await manager.disconnect_all()
    else:
        _default_logger.error("Failed to connect to glasses.")


if __name__ == "__main__":
    asyncio.run(main())