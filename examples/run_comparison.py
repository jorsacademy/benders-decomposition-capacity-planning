from __future__ import annotations

import json

from benders_capacity.experiment import comparison_report


if __name__ == "__main__":
    print(json.dumps(comparison_report(), indent=2, sort_keys=True))
