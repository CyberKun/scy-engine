# Scy Engine

A lightweight Python Go (Baduk) engine with a clean web interface.

## The Why

I had a Go engine running, but testing it purely in the terminal was painful. Playing against text grids gets old fast. Instead of dealing with heavy frontend frameworks, I wrapped the engine in a simple Flask server and a Vanilla JS frontend. No React, no build steps, just pure HTML/JS and a Python backend doing the heavy lifting.

## Features (TL;DR)

- **Human vs AI:** Play direct matches against the engine.
- **Analysis Mode:** Hover over the board to see win rates and MCTS visit counts for specific moves.
- **Live MCTS Visualizer:** Watch the engine think in real-time as it builds its search tree and evaluates positions.

## Quick Start

You don't need much to get this running.

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/scy-engine.git
cd scy-engine
```

2. Install dependencies (it's basically just Flask)
```bash
python -m venv venv
venv\Scripts\activate  # or `source venv/bin/activate` on mac/linux
pip install -r requirements.txt
```

3. Run the server
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

## Tech Stack

- **Backend:** Python 3, Flask
- **Engine:** Custom Monte Carlo Tree Search (MCTS) implementation in raw Python
- **Frontend:** Vanilla HTML/CSS/JS (Zero build tools)
