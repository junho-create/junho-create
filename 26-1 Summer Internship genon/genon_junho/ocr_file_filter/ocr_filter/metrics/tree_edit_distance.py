"""Zhang-Shasha (1989) 순서있는 트리 편집거리. teds.py 가 사용하는 범용 알고리즘 —
외부 의존성(apted/zss 등) 없이 순수 파이썬으로 구현. 노드 종류에 상관없이
`children: list` + 라벨 비교 함수만 있으면 어떤 트리에도 쓸 수 있다.
"""

from __future__ import annotations

from typing import Callable, TypeVar

N = TypeVar("N")


def _postorder(root: N, get_children: Callable[[N], list]) -> list[N]:
    order: list[N] = []

    def visit(node: N) -> None:
        for c in get_children(node):
            visit(c)
        order.append(node)

    visit(root)
    return order


def _leftmost_leaf_indices(order: list[N], get_children: Callable[[N], list]) -> list[int]:
    """order[i] (0-indexed) 의 leftmost-leaf-descendant 를 postorder 인덱스(0-indexed)로."""
    pos = {id(node): i for i, node in enumerate(order)}
    cache: dict[int, int] = {}

    def leftmost(node: N) -> int:
        if id(node) in cache:
            return cache[id(node)]
        children = get_children(node)
        result = leftmost(children[0]) if children else pos[id(node)]
        cache[id(node)] = result
        return result

    return [leftmost(node) for node in order]


def _keyroots(l: list[int]) -> list[int]:
    """postorder 인덱스(0-indexed) 중 keyroot 만: 같은 l 값을 가진 것 중 가장 큰 인덱스."""
    best: dict[int, int] = {}
    for i, li in enumerate(l):
        best[li] = i  # 오름차순으로 훑으므로 마지막에 남는 게 최댓값
    return sorted(best.values())


def tree_edit_distance(
    root_a: N, root_b: N,
    get_children: Callable[[N], list],
    label_eq: Callable[[N, N], bool] | None = None,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    sub_cost: float = 1.0,
    rename_cost: Callable[[N, N], float] | None = None,
) -> float:
    """순서있는 트리 A, B 사이 편집거리 (삽입/삭제/치환).

    치환비용은 둘 중 하나로 준다:
      - `label_eq`: True/False 만(같으면 0, 다르면 sub_cost) — 표 TEDS처럼 "정확히 같다/다르다"만
        따질 때. (기존 API, 하위호환)
      - `rename_cost`: 0~1 연속값을 직접 계산하는 콜백 — 일반 HTML 트리 TEDS처럼 "태그는 같은데
        텍스트만 약간 다르다" 같은 걸 부분점수로 반영하고 싶을 때 (`generic_teds.py` 가 사용).
    둘 다 안 주면 에러."""
    if label_eq is None and rename_cost is None:
        raise ValueError("label_eq 또는 rename_cost 둘 중 하나는 반드시 줘야 함")

    def _cost(a: N, b: N) -> float:
        if rename_cost is not None:
            return rename_cost(a, b)
        return 0.0 if label_eq(a, b) else sub_cost

    a_order = _postorder(root_a, get_children)
    b_order = _postorder(root_b, get_children)
    n, m = len(a_order), len(b_order)
    if n == 0:
        return m * insert_cost
    if m == 0:
        return n * delete_cost

    a_l = _leftmost_leaf_indices(a_order, get_children)
    b_l = _leftmost_leaf_indices(b_order, get_children)
    a_keyroots = _keyroots(a_l)
    b_keyroots = _keyroots(b_l)

    # treedists[i][j] = 서브트리 a_order[i] vs b_order[j] 의 트리편집거리 (0-indexed)
    treedists = [[0.0] * m for _ in range(n)]

    for i in a_keyroots:
        li = a_l[i]
        for j in b_keyroots:
            lj = b_l[j]
            width_a = i - li + 2   # forest 크기(0=빈 forest 포함)
            width_b = j - lj + 2
            fd = [[0.0] * width_b for _ in range(width_a)]

            for x in range(1, width_a):
                fd[x][0] = fd[x - 1][0] + delete_cost
            for y in range(1, width_b):
                fd[0][y] = fd[0][y - 1] + insert_cost

            for x in range(1, width_a):
                ii = li + x - 1
                for y in range(1, width_b):
                    jj = lj + y - 1
                    if a_l[ii] == li and b_l[jj] == lj:
                        cost = _cost(a_order[ii], b_order[jj])
                        fd[x][y] = min(
                            fd[x - 1][y] + delete_cost,
                            fd[x][y - 1] + insert_cost,
                            fd[x - 1][y - 1] + cost,
                        )
                        treedists[ii][jj] = fd[x][y]
                    else:
                        p = a_l[ii] - li
                        q = b_l[jj] - lj
                        fd[x][y] = min(
                            fd[x - 1][y] + delete_cost,
                            fd[x][y - 1] + insert_cost,
                            fd[p][q] + treedists[ii][jj],
                        )

    return treedists[n - 1][m - 1]
