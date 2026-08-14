# Demo notebooks

Runnable notebooks showcasing screamer, using only the public API. Notebooks
01-10 are self-contained with seeded synthetic data. The microstructure
notebooks (11-13) read a small committed real-data slice from `data/` (six hours
of Deribit BTC- and ETH-perpetual trades) for the flow sections; the book
operators in notebook 13 use an illustrative synthetic L1 book, since a trade
tape carries no quotes. The backtest notebooks (14-15) reuse the same real-data
slice: 14 marks a position signal to bars, and 15 drives the event-driven engines
(OHLC, tape, and a synthetic L1 book) from one tape. Everything runs offline with
no downloads.

## Run / verify

```bash
make notebooks     # builds the lib + bindings, then executes every notebook
# or, against an already-built install:
poetry run pytest --nbmake docs/notebooks/
```

Open any `.ipynb` in Jupyter to read/edit. A notebook that computes an equality
(e.g. batch == stream) asserts it in-cell, so a regression fails the run above.

The commodity research notebook (`21-corn-futures-indicator-study.ipynb`) is
different: Bloomberg data is licensed and must remain local. It includes a
clearly labelled deterministic demo fixture for rendering; set `SOURCE` to
`bloomberg` or `local` for an actual study.
