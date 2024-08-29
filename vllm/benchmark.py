from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from time import perf_counter


# Summary viz config
TOPK = 6
SORT_BY = "total"
DOUBLE_SPACE = True

CHARS = 7
RANGE_DELIM = "|"
TIME_UNIT_CONVERSIONS = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
}


def _format_float(f: float) -> str:
    s = f"%.3f" % f
    shift = CHARS - len(s)
    return " " * shift + s


def _format_int(i: int) -> str:
    s = str(int(i))
    buffer = 1
    shift = CHARS - len(s) - buffer
    return " " * shift + s


class TimeRange:

    def __init__(self, unit: str = "s", name: str = None) -> None:
        assert unit in TIME_UNIT_CONVERSIONS, f"unsupported unit {unit} for TimeRange {name}"
        self.name = name
        self.unit = unit
        self.times: List[int] = []
        self.start_time: datetime = None

    @staticmethod
    def summary_header(topk: int) -> str:
        return " total    count    max      avg   |    " + "        ".join(map(str, range(topk)))

    def start(self) -> None:
        assert self.start_time == None, f"TimeRange {self.name} already started"
        self.start_time = perf_counter()

    def end(self, time=None) -> None:
        if time is None:
            time = perf_counter()
        self.times.append((time - self.start_time))
        self.start_time = None

    def avg(self) -> int:
        return sum(self.times) / len(self.times)

    def max(self) -> int:
        return max(self.times)

    def count(self) -> int:
        return len(self.times)

    def total(self) -> int:
        return sum(self.times)

    def topk(self, k: int) -> List[int]:
        return list(sorted(self.times, reverse=True))[:k]

    def summarize(self, topk: int) -> str:
        text =  "  ".join(
            [(_format_int if agg == "count" else _format_float)(getattr(self, agg)()) for agg in
             ["total", "count", "max", "avg"]]
        )
        text += " | " + "  ".join(
            [_format_float(f) for f in
             list(sorted(self.times, reverse=True))[:topk]]
        )
        return text


class EnterableRange:

    def __init__(self, rng: TimeRange):
        self.range = rng

    def __enter__(self):
        self.range.start()

    def __exit__(self, *_, **__):
        self.range.end()


class EnterableConditional:

    def __init__(self, benchmarker: "Benchmarker", do_benchmark: bool):
        self.do_benchmark = do_benchmark
        self.benchmarker = benchmarker
        self.reenable_on_exit = not benchmarker.disabled

    def __enter__(self):
        if not self.do_benchmark:
            self.benchmarker.disabled = True

    def __exit__(self, *_, **__):
        if self.reenable_on_exit:
            self.benchmarker.disabled = False


class NullEnterable:

    def __enter__(self):
        pass

    def __exit__(self, *_, **__):
        pass


class Benchmarker:

    def __init__(self, time_range_unit: str = "s"):
        self.time_range_unit = time_range_unit
        self.ranges: Dict[str, TimeRange] = {}
        self.range_head: List[str] = []
        self.range_hierarchy: Dict[str, Set[str]] = {}
        self.outer_ranges: Set[str] = set()
        self.disabled = False
        self.depth = 0

    def _contextualize_name(
        self,
        name: str,
        range_head: Optional[List[str]] = None,
    ) -> str:
        range_head = self.range_head if range_head is None else range_head
        return RANGE_DELIM.join(range_head + [name])

    def _walk_in(self, name: str) -> None:
        self.depth += 1
        # print(f"walking in: {name} ({self.depth})")
        if name not in self.range_hierarchy:
            self.range_hierarchy[name] = set()
        if len(self.range_head) == 0:
            self.outer_ranges.add(name)
        else:
            self.range_hierarchy[self.range_head[-1]].add(name)
        self.range_head.append(name)

    def _walk_out(self) -> None:
        self.depth -= 1
        # print(f"walking out: {self.range_head[-1]} ({self.depth})")
        self.range_head.pop(-1)

    def time(self, name: str) -> EnterableRange:
        if self.disabled:
            return NullEnterable()
        if name not in self.ranges:
            self.ranges[name] = TimeRange(self.time_range_unit, name)
        return EnterableRange(self.ranges[name])

    def start_range(self, name: str) -> None:
        if not self.disabled:
            cont_name = self._contextualize_name(name)
            self._walk_in(name)
            name = cont_name
            if name not in self.ranges:
                self.ranges[name] = TimeRange(self.time_range_unit, name)
            self.ranges[name].start()

    def end_range(self, name: str) -> None:
        end_time = perf_counter()
        if not self.disabled:
            self._walk_out()
            name = self._contextualize_name(name)
            assert name in self.ranges, f"no TimeRange found for {name}"
            self.ranges[name].end(time=end_time)

    def _summarize_recursive(
        self,
        range_head: List[str],
        topk: int,
        sort_by: str,
        depth: int,
    ) -> List[Tuple[int, str, str]]:
        if range_head:
            children = self.range_hierarchy[range_head[-1]]
        else:
            children = self.outer_ranges
        if not children:
            return []
        sorted_children = sorted(
            map(lambda c: (c, self._contextualize_name(c, range_head)), children),
            key=lambda c: getattr(self.ranges[c[1]], sort_by)(),
            reverse=True,
        )
        return_list = []
        for i, (c, c_cont) in enumerate(sorted_children):
            rng = self.ranges[c_cont]
            is_last = i == len(sorted_children) - 1
            return_list.append((depth, c, rng.summarize(topk), is_last))
            return_list.extend(
                self._summarize_recursive(range_head + [c], topk, sort_by, depth + 1)
            )
        return return_list

    def summarize(self):
        summary_list = self._summarize_recursive([], TOPK, SORT_BY, 0)
        names = []
        continue_for_depths = set()
        for depth, name, _, is_last in summary_list:
            if depth > 0:
                continue_str = "".join(
                    ["| " if i in continue_for_depths else "  " for i in range(depth - 1)]
                )
                tick_str = "'-" if is_last else "|-"
                names.append(continue_str + tick_str + name)
                continue_for_depths.add(depth - 1)
                if is_last:
                    continue_for_depths.remove(depth - 1)
            else:
                names.append(name)
        max_offset = max([len(n) for n in names])
        buffer = 2
        print("\nLatency Breakdown:")
        print(" " * (max_offset + buffer) + TimeRange.summary_header(TOPK))
        for name, (_, _, summary, _) in zip(names, summary_list):
            s = f"{name}{' ' * buffer}{' ' * (max_offset - len(name))}{summary}"
            if DOUBLE_SPACE:
                print("".join(["|" if char in ["|", "'"] else " " for char in s]))
            print(s)


    def summarize_old(self) -> None:
        print("Aggregated times:")
        max_len_name = max(map(len, self.ranges.keys())) + 2  # buffer
        n_times = 6
        times_header = "\t".join(map(str, range(n_times)))
        print(f"range_name{' ' * (max_len_name - 10)}avg\tmax\t |\t{times_header}")
        for name, rng in self.ranges.items():
            avg = "%.3f" % rng.avg()
            max_ = "%.3f" % rng.max()
            shift = max_len_name - len(name)
            formatted_times = "\t".join(map(lambda t: "%.3f" % t, list(sorted(rng.times, reverse=True))[:6]))
            print(f"{name}{' ' * shift}{avg}\t{max_}\t |\t{formatted_times}")

    def conditional(self, do_benchmark: bool) -> EnterableConditional:
        return EnterableConditional(self, do_benchmark)

    def wrap(self):
        def wrapper(fn):
            fn_name = f"{fn.__module__}.{fn.__qualname__}"
            def fn_(*args, **kwargs):
                if self.disabled:
                    return fn(*args, **kwargs)
                self.start_range(fn_name)
                out = fn(*args, **kwargs)
                self.end_range(fn_name)
                return out
            return fn_
        return wrapper

    def wrap_if(self, **conditions):
        def wrapper(fn):
            fn_name = f"{fn.__module__}.{fn.__qualname__}"
            def fn_(*args, **kwargs):
                if self.disabled:
                    return fn(*args, **kwargs)
                for k, v in conditions.items():
                    if kwargs.get(k) != v:
                        self.disabled = True
                        out = fn(*args, **kwargs)
                        self.disabled = False
                        return out
                self.start_range(fn_name)
                out = fn(*args, **kwargs)
                self.end_range(fn_name)
                return out
            return fn_
        return wrapper


BENCHMARKER = Benchmarker("ms")
