from __future__ import annotations

from src.data.non_price_data import collect_news_data


def main() -> None:
    paths = collect_news_data()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

