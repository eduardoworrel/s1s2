"""s1s2-heads from-<format> ... (each subcommand: see formats.py). `register(sub)` adds the same subcommands to
another argparse CLI (the s1s2-convert CLI of converter/, once both branches meet)."""
import argparse

from .formats import register


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="s1s2-heads", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
