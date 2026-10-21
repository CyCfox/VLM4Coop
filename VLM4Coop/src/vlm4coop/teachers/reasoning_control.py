"""Bound spending when a provider stops honouring the requested OFF setting."""
from collections import deque
import threading


class ReasoningControlError(RuntimeError):
    pass


class ReasoningGuard:
    def __init__(self, policy):
        self.startup = int(policy['startup_clean_responses'])
        self.window = int(policy['window'])
        self.allowed = int(policy['max_reasoning_responses'])
        self.max_in_flight = int(policy['max_in_flight'])
        if not (self.startup > 0 and self.window > self.allowed >= 0 and self.max_in_flight > 0):
            raise ValueError('Invalid reasoning guard policy')
        self.history = deque(maxlen=self.window)
        self.clean_startup = 0
        self.error = None
        self.condition = threading.Condition()
        self.slots = threading.BoundedSemaphore(self.max_in_flight)
        self.startup_in_flight = False

    def enter(self):
        self.slots.acquire()
        with self.condition:
            while self.clean_startup < self.startup and self.startup_in_flight and not self.error:
                self.condition.wait()
            if self.error:
                self.slots.release()
                raise ReasoningControlError(self.error)
            if self.clean_startup < self.startup:
                self.startup_in_flight = True

    def observe(self, evidence):
        with self.condition:
            thought = evidence == 'returned_reasoning'
            self.history.append(thought)
            if self.clean_startup < self.startup:
                if thought:
                    self.error = 'Provider returned reasoning during OFF startup verification; no automatic retry'
                else:
                    self.clean_startup += 1
            elif sum(self.history) > self.allowed:
                self.error = f'Provider reasoning exceeded {self.allowed} response(s) in the last {self.window}; API work stopped'
            if self.error:
                self.condition.notify_all()
                raise ReasoningControlError(self.error)

    def leave(self):
        with self.condition:
            self.startup_in_flight = False
            self.condition.notify_all()
        self.slots.release()
