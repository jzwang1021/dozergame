"""预计算固定地形回放，并把回放与 Three.js 直接嵌入 index.html。"""

import json
from math import cos, pi, sin
from pathlib import Path
import re
import sys

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
from grid_game_v1 import solve_transport

SIZE = 16
UNITS = 1000
CAPACITY = 2 * UNITS
PRESETS = [(20261125, 0.9, "缓坡浅洼"), (20261126, 1.25, "交错丘谷"), (20261127, 1.75, "起伏山脊")]


def terrain(seed, roughness):
    state = seed
    noise = []
    for _ in range(4 + SIZE**2):
        state = (1664525 * state + 1013904223) % 2**32
        noise.append(state / 2**32)
    phases = [value * 2 * pi for value in noise[:4]]
    noise = np.asarray(noise[4:]).reshape(SIZE, SIZE)
    border = np.pad(noise, 1, mode="edge")
    smooth = (4 * noise + border[1:-1, :-2] + border[1:-1, 2:]
              + border[:-2, 1:-1] + border[2:, 1:-1]) / 8
    offsets = []
    for i, value in enumerate(smooth.flat):
        row, column = divmod(i, SIZE)
        x, z = column / (SIZE - 1) * pi, row / (SIZE - 1) * pi
        offsets.append(roughness * (.86 * sin(2*x + phases[0]) * cos(2*z + phases[1])
                       + .55 * sin(1.5*x + 2.3*z + phases[2])
                       + .29 * cos(3.8*x - 2.5*z + phases[3]) + .22 * (value - .5)))
    offsets = np.asarray(offsets)
    scaled = (offsets - offsets.mean()) * UNITS
    units = np.floor(scaled).astype(int)
    missing = -int(units.sum())
    units[np.argsort(units - scaled, kind="stable")[:missing]] += 1
    assert units.sum() == 0
    return units


def distance(a, b):
    ay, ax = divmod(a, SIZE)
    by, bx = divmod(b, SIZE)
    return abs(ay - by) + abs(ax - bx)


def greedy_trips(units):
    cut = {i: int(q) for i, q in enumerate(units) if q > 0}
    fill = {i: int(-q) for i, q in enumerate(units) if q < 0}
    position = 0
    while cut:
        source = min(cut, key=lambda i: distance(position, i))
        sink = min(fill, key=lambda i: distance(source, i))
        amount = min(cut[source], fill[sink], CAPACITY)
        yield source, sink, amount
        for stock, cell in ((cut, source), (fill, sink)):
            stock[cell] -= amount
            if not stock[cell]:
                del stock[cell]
        position = sink


def optimized_trips(units, solver):
    amounts = units.reshape(SIZE, SIZE) / UNITS
    trips, _ = solve_transport(np.maximum(amounts, 0), np.maximum(-amounts, 0),
                               1, 10_000, 30, solver)
    pending = [(int(a), int(b), round(q * UNITS)) for a, b, q in trips]
    position = 0
    while pending:
        index = min(range(len(pending)), key=lambda i: distance(position, pending[i][0]))
        source, sink, quantity = pending[index]
        amount = min(quantity, CAPACITY)
        yield source, sink, amount
        if quantity > amount:
            pending[index] = source, sink, quantity - amount
        else:
            pending.pop(index)
        position = sink


def replay(units, trips):
    actions = []
    stock = units.copy()
    position = work = empty = steps = 0

    def move_to(destination, load):
        nonlocal position, work, empty, steps
        row, column = divmod(position, SIZE)
        end_row, end_column = divmod(destination, SIZE)
        for difference, horizontal in ((end_column - column, True), (end_row - row, False)):
            direction = 1 if difference > 0 else -1
            actions.extend([["move", direction, 0] if horizontal else ["move", 0, direction]]
                           * abs(difference))
        count = distance(position, destination)
        steps += count
        work += load * count
        empty += count if load == 0 else 0
        position = destination

    for source, sink, amount in trips:
        if not 0 < amount <= min(CAPACITY, stock[source], -stock[sink]):
            raise ValueError("回放工单超出格点库存或铲刀容量")
        move_to(source, 0)
        actions.append(["transfer", amount / UNITS])
        stock[source] -= amount
        move_to(sink, amount)
        actions.append(["transfer", -amount / UNITS])
        stock[sink] += amount
    if np.any(stock):
        raise ValueError("回放结束后仍有剩余挖填量")
    return {"actions": actions, "score": {"work": work / UNITS, "empty": empty, "steps": steps,
                                           "progress": 100}}


def generate():
    scenarios = []
    for seed, roughness, label in PRESETS:
        units = terrain(seed, roughness)
        algorithms = {"greedy": replay(units, greedy_trips(units))}
        for solver in ("highs", "ortools"):
            algorithms[solver] = replay(units, optimized_trips(units, solver))
        if algorithms["highs"]["score"]["work"] != algorithms["ortools"]["score"]["work"]:
            raise ValueError(f"种子 {seed} 的最优搬运功互验失败")
        scenarios.append({"seed": seed, "label": label, "size": SIZE, "roughness": roughness,
                          "target": 3, "capacity": CAPACITY / UNITS,
                          "heights": (3 + units / UNITS).tolist(), "algorithms": algorithms})
    payload = json.dumps({"schemaVersion": 1, "scenarios": scenarios}, ensure_ascii=False, separators=(",", ":"))
    (DIRECTORY / "algorithms.json").write_text(payload + "\n", encoding="utf-8")
    path = DIRECTORY / "index.html"
    html = path.read_text(encoding="utf-8")
    for identifier, content in (
        ("three-library", (DIRECTORY / "three.r160.js").read_text(encoding="utf-8")),
        ("algorithm-data", payload),
    ):
        pattern = rf'(<script id="{identifier}"[^>]*>)[\s\S]*?(</script>)'
        if "</script" in content.lower():
            raise ValueError(f"{identifier} 包含未转义的 script 结束标签")
        html, count = re.subn(pattern, lambda match: match[1] + "\n" + content.rstrip()
                             + "\n" + match[2], html, count=1)
        if count != 1:
            raise ValueError(f"找不到 {identifier} 嵌入位置")
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    generate()
