import heapq

# Provides type clarity without the overhead of Python objects
# from something like a dataclass. Stores (time, neuron_idx)
SpikeEvent = tuple[float, int]


class EventQueue:
    """
    Min-heap priority queue for spike events

    initialize_heap should be called first to load the queue with initial
    events and heapify it, which is more efficient than pushing events one
    at a time. After that, push and pop can be used to manage the queue.

    heapq naturally supports tuples where the first element is the priority
    (time in this case).
    """

    def __init__(self):
        self._heap = []

    def initialize_heap(self, events: list[SpikeEvent]) -> None:
        self._heap.extend(events)
        heapq.heapify(self._heap)

    def push(self, event: SpikeEvent) -> None:
        heapq.heappush(self._heap, event)

    def pop(self) -> SpikeEvent:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
