import heapq

# Provides type clarity without the overhead of Python objects
# from something like a dataclass. Represents time of spike
# arrival, the index of the target neuron (where the spike is arriving),
# and the weight of the connection between the spiking neuron
# and the target neuron:
# (arrival_time, neuron_idx, weight)
SpikeArrivalEvent = tuple[float, int, float]


class EventQueue:
    """
    Min-heap priority queue for spike arrival events

    initialize_heap should be called first to load the queue with initial
    events and heapify it, which is more efficient than pushing events one
    at a time. After that, push and pop can be used to manage the queue.

    heapq naturally supports tuples where the first element is the priority
    (time in this case).
    """

    def __init__(self):
        self._heap = []

    def initialize_heap(self, events: list[SpikeArrivalEvent]) -> None:
        self._heap.extend(events)
        heapq.heapify(self._heap)

    def push(self, event: SpikeArrivalEvent) -> None:
        heapq.heappush(self._heap, event)

    def pop(self) -> SpikeArrivalEvent:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
