from __future__ import annotations

from src.data.price_data import collect_price_data


def main() -> None:
    path = collect_price_data()
    print(path)


if __name__ == "__main__":
    main()
