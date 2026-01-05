# main.py
from __future__ import annotations

import argparse

from demo import (
    demo_equity_config,
    demo_girr_cfg_bonds,
    demo_girr_cfg_swaps,
    demo_market,
    demo_portfolio,
)
from engine import FRTBEngine
from portfolio import load_portfolio_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=str, default="", help="CSV portfolio path (optional)")
    ap.add_argument("--quiet", action="store_true", help="moins de prints")
    args = ap.parse_args()

    port = load_portfolio_csv(args.portfolio) if args.portfolio else demo_portfolio()

    mkt, bond_override = demo_market()
    engine = FRTBEngine(
        market=mkt,
        equity_cfg=demo_equity_config(),
        girr_cfg_swaps=demo_girr_cfg_swaps(),
        girr_cfg_bonds=demo_girr_cfg_bonds(),
    )

    engine.run(
        port=port,
        bond_curves_override=bond_override,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
