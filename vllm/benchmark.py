from typing import List, Dict, Set
from datetime import datetime


TIME_UNIT_CONVERSIONS = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
}


class TimeRange:
    
    def __init__(self, unit: str = "s", name: str = None) -> None:
        assert unit in TIME_UNIT_CONVERSIONS, f"unsupported unit {unit} for TimeRange {name}"
        self.name = name
        self.unit = unit
        self.times: List[int] = []
        self.start_time: datetime = None

    def start(self) -> None:
        assert self.start_time == None, f"TimeRange {self.name} already started"
        self.start_time = datetime.now()

    def end(self) -> None:
        self.times.append((datetime.now() - self.start_time).total_seconds()
                          * TIME_UNIT_CONVERSIONS[self.unit])
        self.start_time = None

    def aggregate_time(self) -> int:
        return sum(self.times) / len(self.times)
    

class EnterableRange:

    def __init__(self, rng: TimeRange):
        self.range = rng

    def __enter__(self):
        self.range.start()

    def __exit__(self, *_, **__):
        self.range.end()


class Benchmarker:

    def __init__(self, time_range_unit: str = "s"):
        self.time_range_unit = time_range_unit
        self.ranges: Dict[str, TimeRange] = {}
        self.range_head: str = None
        self.range_hierarchy: Dict[str, Set[str]] = {}
        self.outer_ranges: List[str] = []

    def _record_hierarchy(self, name: str) -> None:
        if name not in self.range_hierarchy:
            self.range_hierarchy[name] = set()
        if self.range_head is None:
            self.outer_ranges.append(name)
        else:
            self.range_hierarchy[self.range_head].add(name)
        self.range_head = name

    def time(self, name: str) -> EnterableRange:
        if name not in self.ranges:
            self.ranges[name] = TimeRange(self.time_range_unit, name)
        return EnterableRange(self.ranges[name])

    def start_range(self, name: str) -> None:
        if name not in self.ranges:
            self.ranges[name] = TimeRange(self.time_range_unit, name)
        self.ranges[name].start()

    def end_range(self, name: str) -> None:
        assert name in self.ranges, f"no TimeRange found for {name}"
        self.ranges[name].end()

    def summarize(self) -> None:
        print("Aggregated times:")
        for name, rng in self.ranges.items():
            print(f"{name}: {rng.aggregate_time()}{self.time_range_unit}")


BENCHMARKER = Benchmarker("ms")
