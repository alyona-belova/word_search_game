# Змейка — Word Search Game

A word search puzzle game. Each level challenges you to find hidden words snaking through a 12×12 letter grid, all sharing a common theme letter.

![Game Logo](game_logo.png)

---

## How to Play

1. **Find the theme letter** — every level has one highlighted letter that all target words contain
2. **Drag to select** — trace a path across the grid to spell out a word
3. **Words snake** — words can twist, turn, and change direction, like a snake
4. **Earn hints** — find 3 bonus words (words from other levels) to unlock a hint that highlights one remaining word

---

## Features

- **10,000+ Russian words** loaded from a curated word list
- **Infinite levels** — each level randomly picks a theme letter and matching words
- **Snaking paths** — words are placed using a DFS pathfinding algorithm with direction bias for natural-looking curves
- **5 color palette** — found words rendered as colored SVG polylines over the grid
- **Hint system** — rewards exploration of bonus words
- **Progress persistence** — game state saved to `localStorage` so you never lose your place
- **First-time tutorial** — a modal walkthrough shown on first visit, always accessible via `?`
- **A/B testing** — randomized UI variants for found-word display (badge vs. list)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | TypeScript (compiled to ES2020) |
| Rendering | Vanilla DOM + SVG overlays |
| Styling | CSS with custom properties, `clamp()`, grid layout |
| Storage | `localStorage` |
| Build | `tsc` |

No frameworks. No bundler. Just TypeScript compiled to a single `game.js`.

---

## Project Structure

```
word_search_game/
├── game.ts               # Core game engine (~900 lines)
├── game.js               # Compiled output
├── index.html            # App shell + Russian UI copy
├── style.css             # Styles, animations, responsive layout
├── game_logo.png         # Logo
├── data/
│   └── words_list_extended.txt   # 10,171 Russian words
├── package.json
└── tsconfig.json
```

---

## Getting Started

**Requirements:** Node.js (for TypeScript compiler)

```bash
# Install dependencies
npm install

# Build once
npm run build

# Watch mode (rebuilds on save)
npm run watch
```

Then open `index.html` in a browser. No server required — the word list is loaded via `fetch`, so you may need a local HTTP server if your browser blocks local file requests:

```bash
npx serve .
# or
python3 -m http.server
```

---

## Architecture Notes

### Grid Generation
A 12×12 grid is populated by placing words one at a time using `placeWordSnaking()`. Each word is threaded through the grid along a path built by a biased DFS: the algorithm prefers to continue in the current direction before trying other directions, producing organic-looking curves.

### Word Selection & Validation
Player selections are tracked as a sequence of `[row, col]` cells. On pointer-up, the selected string is checked against all unfound theme words. Non-theme words that match the word list count toward the hint counter.

### Hint System
Finding 3 extra (non-theme) words unlocks one hint. A hint animates the cells of one remaining theme word, giving the player a nudge without fully solving it.

### Persistence
`GameProgress` is serialized to JSON and stored in `localStorage`. It captures the full grid state, found words, hint counts, current level, and A/B group assignment.
