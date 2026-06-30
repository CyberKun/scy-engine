# Scy Engine

A high-performance Python-based Go (Baduk) engine built around Monte Carlo Tree Search (MCTS), served via a lightweight Flask backend and visualized with Vanilla JS.

## Core Architecture
- **Search Engine**: Custom implementation of Monte Carlo Tree Search (MCTS) utilizing UCB1-Tuned for node selection and optimized fast-random playouts.
- **Backend API**: Flask-based REST and Server-Sent Events (SSE) endpoints for real-time tree evaluation streaming.
- **Frontend**: Zero-dependency Vanilla HTML/JS client rendering dynamic board states and live MCTS analytics.

## Setup & Installation

`ash
git clone <repo-url>
cd scy-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
`

## System Design
The engine models the Go board as an optimized 2D grid supporting fast adjacency operations, flood-fill group tracking, and rigorous ko/suicide detection. Move selection is driven by an asynchronous MCTS pipeline that evaluates thousands of playouts. It streams intermediate search tree statistics - including node visit distributions, win rate confidences, and UCB1 values - via SSE back to the client, providing deep inspection into the adversarial search process.
