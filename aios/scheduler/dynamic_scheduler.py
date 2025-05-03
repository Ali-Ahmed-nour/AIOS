from .base import BaseScheduler
from queue import Queue, Empty
from ..context.simple_context import SimpleContextManager

from aios.hooks.types.llm import LLMRequestQueueGetMessage
from aios.hooks.types.memory import MemoryRequestQueueGetMessage
from aios.hooks.types.tool import ToolRequestQueueGetMessage
from aios.hooks.types.storage import StorageRequestQueueGetMessage

import traceback
import time
from aios.utils.logger import SchedulerLogger
from threading import Thread

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DynamicScheduler(BaseScheduler):
    """
    Dynamic scheduler implementation that gives each task a variable time slice
    based on its priority and task size.

    This scheduler ensures efficient resource allocation by dynamically adjusting
    the time slice for each task.

    Example:
        ```python
        scheduler = DynamicScheduler(
            ...
        )
        scheduler.start()
        ```
    """

    def __init__(self, *args, base_time_slice: float = 1, max_waiting_time: float = 30, max_task_size: int = 100, **kwargs):
        """
        Initialize the Dynamic Scheduler.

        Args:
            *args: Arguments passed to BaseScheduler
            base_time_slice: Base time slice for tasks in seconds
            max_waiting_time: Maximum waiting time for tasks (used in priority calculation)
            max_task_size: Maximum task size (used in priority calculation)
            **kwargs: Keyword arguments passed to BaseScheduler
        """
        super().__init__(*args, **kwargs)
        self.base_time_slice = base_time_slice
        self.max_waiting_time = max_waiting_time
        self.max_task_size = max_task_size
        self.context_manager = SimpleContextManager()

    def calculate_dynamic_priority(self, syscall: Any) -> float:
        """
        Calculate dynamic priority based on waiting time and task size.

        Args:
            syscall: The system call containing task information.

        Returns:
            float: Updated priority for the syscall.
        """
        # Normalize factors
        waiting_time_factor = min(1, syscall.waiting_time / self.max_waiting_time)
        task_size_factor = syscall.task_size / self.max_task_size

        # Calculate final priority
        priority = syscall.base_priority + (waiting_time_factor * 0.5) \
                   + (task_size_factor * 0.5)

        return max(1, priority)  # Ensure priority is at least 1

    def calculate_dynamic_time_slice(self, syscall: Any) -> float:
        """
        Calculate the dynamic time slice for a syscall based on its priority.

        Args:
            syscall: The system call to calculate the time slice for

        Returns:
            float: Dynamically calculated time slice in seconds
        """
        # Calculate Priority Factor
        priority = self.calculate_dynamic_priority(syscall)
        priority_factor = priority / 10

        # Ensure the time slice is not too small or too large
        dynamic_time_slice = max(0.5, self.base_time_slice * priority_factor)
        return dynamic_time_slice

    def _execute_syscall(
        self, 
        syscall: Any,
        executor: Any,
        syscall_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a system call with dynamic time slice enforcement.

        Args:
            syscall: The system call to execute
            executor: Function to execute the syscall
            syscall_type: Type of the syscall for logging

        Returns:
            Optional[Dict[str, Any]]: Response from the syscall execution
        """
        try:
            # Calculate the dynamic time slice
            time_slice = self.calculate_dynamic_time_slice(syscall)

            syscall.set_time_limit(time_slice)
            syscall.set_status("executing")
            self.logger.log(
                f"{syscall.agent_name} is executing {syscall_type} syscall with time slice {time_slice:.2f}s.\n",
                "executing"
            )
            syscall.set_start_time(time.time())

            response = executor(syscall)

            syscall.set_response(response)

            if response.finished:
                syscall.set_status("done")
                log_status = "done"
            else:
                syscall.set_status("suspending")
                log_status = "suspending"

            syscall.set_end_time(time.time())

            syscall.event.set()

            self.logger.log(
                f"{syscall_type} syscall for {syscall.agent_name} is {log_status}. "
                f"Thread ID: {syscall.get_pid()}\n",
                log_status
            )

            return response

        except Exception as e:
            logger.error(f"Error executing {syscall_type} syscall: {str(e)}")
            traceback.print_exc()
            return None

    def process_llm_requests(self) -> None:
        """
        Process LLM requests with dynamic time slicing.
        """
        while self.active:
            try:
                llm_syscall = self.get_llm_syscall()
                self._execute_syscall(llm_syscall, self.llm.execute_llm_syscall, "LLM")
            except Empty:
                pass

    def process_memory_requests(self) -> None:
        """
        Process Memory requests with dynamic time slicing.
        """
        while self.active:
            try:
                memory_syscall = self.get_memory_syscall()
                self._execute_syscall(
                    memory_syscall,
                    self.memory_manager.address_request,
                    "Memory"
                )
            except Empty:
                pass

    def process_storage_requests(self) -> None:
        """
        Process Storage requests with dynamic time slicing.
        """
        while self.active:
            try:
                storage_syscall = self.get_storage_syscall()
                self._execute_syscall(
                    storage_syscall,
                    self.storage_manager.address_request,
                    "Storage"
                )
            except Empty:
                pass

    def process_tool_requests(self) -> None:
        """
        Process Tool requests with dynamic time slicing.
        """
        while self.active:
            try:
                tool_syscall = self.get_tool_syscall()
                self._execute_syscall(
                    tool_syscall,
                    self.tool_manager.address_request,
                    "Tool"
                )
            except Empty:
                pass

    def start(self) -> None:
        """
        Start all request processing threads.
        """
        self.active = True
        self.start_processing_threads([
            self.process_llm_requests,
            self.process_memory_requests,
            self.process_storage_requests,
            self.process_tool_requests
        ])

    def stop(self) -> None:
        """
        Stop all request processing threads.
        """
        self.active = False
        self.stop_processing_threads()