from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


GridKey = Tuple[int, int]


@dataclass
class GlobalCornerIdentity:
    global_row: int
    global_col: int

    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]

    votes: int = 0


class IdentityStore:
    """
    Persistent global identity storage.

    IMPORTANT:
    Local checkerboard indices are NOT persistent semantic identities.
    They may change during recovery / lattice refit / tracking updates.

    Therefore:
        - only global IDs are stored persistently
        - local IDs are frame-local only
    """

    def __init__(self) -> None:
        self._items: List[GlobalCornerIdentity] = []

        # Persistent storage ONLY by global key.
        self._by_global: Dict[GridKey, GlobalCornerIdentity] = {}

    def clear(self) -> None:
        self._items.clear()
        self._by_global.clear()

    def empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> List[GlobalCornerIdentity]:
        return list(self._items)

    def by_global(self) -> Dict[GridKey, GlobalCornerIdentity]:
        return dict(self._by_global)

    def get_global(
        self,
        global_row: int,
        global_col: int,
    ) -> Optional[GlobalCornerIdentity]:
        return self._by_global.get(
            (int(global_row), int(global_col))
        )

    def has_global(
        self,
        global_row: int,
        global_col: int,
    ) -> bool:
        return (
            int(global_row),
            int(global_col),
        ) in self._by_global

    def global_keys(self) -> set[GridKey]:
        return set(self._by_global.keys())

    def replace(
        self,
        identities: Iterable[GlobalCornerIdentity],
    ) -> None:
        self.clear()
        self.merge(identities)

    def merge(
        self,
        identities: Iterable[GlobalCornerIdentity],
    ) -> None:
        """
        Merge ONLY by global key.

        Never persist local checkerboard indices.
        """

        for p in identities:
            global_key = (
                int(p.global_row),
                int(p.global_col),
            )

            existing = self._by_global.get(global_key)

            if existing is None:
                self._by_global[global_key] = p
                continue

            # Keep higher-vote observation.
            if int(p.votes) >= int(existing.votes):
                self._by_global[global_key] = p

        self._items = list(self._by_global.values())
