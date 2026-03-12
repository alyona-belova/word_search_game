async function loadWordsFromFile() {
  try {
    const response = await fetch("data/words_list.txt");
    const text = await response.text();
    return text
      .split("\n")
      .map((word) => word.trim().toUpperCase())
      .filter((word) => word.length > 0);
  } catch (error) {
    console.error("Ошибка загрузки слов:", error);
    return [];
  }
}

let allWords = [];

const RUSSIAN_ALPHABET = [
  "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
];

const LINE_COLORS = [
  "rgba(193,97,74,0.82)",
  "rgba(90,127,106,0.82)",
  "rgba(184,137,42,0.82)",
  "rgba(74,134,168,0.82)",
  "rgba(155,114,216,0.82)",
];

function removePrefixConflicts(words) {
  return words.filter(
    (w) => !words.some((other) => other !== w && (other.startsWith(w) || w.startsWith(other))),
  );
}

class WordSearchGame {
  constructor() {
    this.currentLevel = 1;
    this.grid = [];
    this.words = [];
    this.foundWords = new Set();
    this.selectedCells = [];
    this.isSelecting = false;
    this.gridSize = 12;
    this.placements = new Map();
    this.demoWordShown = false;
    this.levelName = "";
    this.themeLetter = "";
    this.extraWords = [];
    this.foundExtraWords = new Set();
    this.hintsUsed = 0;
    this.extraWordsFoundCount = 0;
    this.wordStartCells = new Map();
    this.wordPaths = new Map();
    this.hintCells = new Set();

    this.abGroup = localStorage.getItem("abGroup");
    if (!this.abGroup) {
      this.abGroup = Math.random() < 0.5 ? "A" : "B";
      localStorage.setItem("abGroup", this.abGroup);
    }

    this.init();
  }

  async init() {
    allWords = await loadWordsFromFile();
    const loaded = this.loadProgress();
    if (!loaded) {
      this.loadLevel();
    }
    this.setupEventListeners();
  }

  saveProgress() {
    try {
      const progress = {
        currentLevel: this.currentLevel,
        grid: this.grid,
        words: this.words,
        foundWords: Array.from(this.foundWords),
        levelName: this.levelName,
        hintsUsed: this.hintsUsed,
        extraWordsFoundCount: this.extraWordsFoundCount,
        extraWords: this.extraWords,
        foundExtraWords: Array.from(this.foundExtraWords),
        themeLetter: this.themeLetter,
        wordPaths: Array.from(this.wordPaths.entries()),
      };
      localStorage.setItem("wordSearchProgress", JSON.stringify(progress));
    } catch (e) {
      console.error("Ошибка сохранения:", e);
    }
  }

  loadProgress() {
    const saved = localStorage.getItem("wordSearchProgress");
    if (!saved) return false;
    try {
      const progress = JSON.parse(saved);
      this.currentLevel = progress.currentLevel || 1;
      if (progress.grid && progress.words) {
        this.grid = progress.grid;
        this.words = progress.words;
        this.levelName = progress.levelName || "";
        this.foundWords = new Set(progress.foundWords || []);
        this.hintsUsed = progress.hintsUsed || 0;
        this.extraWordsFoundCount = progress.extraWordsFoundCount || 0;
        this.extraWords = progress.extraWords || [];
        this.foundExtraWords = new Set(progress.foundExtraWords || []);
        this.themeLetter = progress.themeLetter || "";
        this.wordPaths = new Map(
          (progress.wordPaths || []).map(([k, v]) => [k, v])
        );
        this.updateThemeDisplay({ name: this.levelName });
        this.rebuildPlacements();
        this.buildGrid();
        this.render();
        if (this.foundWords.size === this.words.length) {
          document.getElementById("levelCompleteMessage").style.display = "block";
        } else {
          document.getElementById("levelCompleteMessage").style.display = "none";
        }
        return true;
      }
    } catch (e) {
      console.error("Ошибка загрузки прогресса:", e);
    }
    return false;
  }

  resetProgress() {
    localStorage.removeItem("wordSearchProgress");
    this.currentLevel = 1;
    this.loadLevel();
  }

  updateGridSizeVariable() {
    document.documentElement.style.setProperty("--grid-size", this.gridSize);
  }

  setupEventListeners() {
    document.addEventListener("pointerup", () => this.stopSelection());
    document.addEventListener("pointercancel", () => this.stopSelection());

    const gridEl = document.getElementById("grid");
    gridEl.addEventListener("pointermove", (e) => {
      if (!this.isSelecting) return;
      const cell = document.elementFromPoint(e.clientX, e.clientY)?.closest(".grid-cell");
      if (!cell) return;
      this.addToSelection(parseInt(cell.dataset.row), parseInt(cell.dataset.col));
    });

    document.getElementById("nextLevelBtn").addEventListener("click", () => this.nextLevel());
  }

  nextLevel() {
    this.currentLevel++;
    this.extraWords = [];
    this.loadLevel();
    this.saveProgress();
  }

  loadLevel(attempt = 0) {
    if (attempt > 50) {
      console.error("Не удалось подобрать букву с достаточным количеством слов");
      return;
    }

    this.hintsUsed = 0;
    this.extraWordsFoundCount = 0;
    this.extraWords = [];
    this.foundExtraWords = new Set();
    this.wordStartCells.clear();
    this.wordPaths.clear();
    this.hintCells = new Set();

    const randomLetter = RUSSIAN_ALPHABET[Math.floor(Math.random() * RUSSIAN_ALPHABET.length)];
    const wordsWithLetter = allWords.filter(
      (word) => word.includes(randomLetter) && word.length <= this.gridSize - 1,
    );

    this.themeLetter = randomLetter;

    if (wordsWithLetter.length < 8) {
      return this.loadLevel(attempt + 1);
    }

    const short = wordsWithLetter.filter((w) => w.length >= 4 && w.length <= 5);
    const medium = wordsWithLetter.filter((w) => w.length >= 6 && w.length <= 7);
    const long = wordsWithLetter.filter((w) => w.length >= 8);

    function pickRandom(arr, count) {
      return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
    }

    this.words = [
      ...pickRandom(short, 2),
      ...pickRandom(medium, 3),
      ...pickRandom(long, 3),
    ];

    this.words = removePrefixConflicts(this.words).sort((a, b) => b.length - a.length);

    this.foundWords.clear();
    this.selectedCells = [];
    this.placements.clear();

    this.levelName = `Буква "${randomLetter}"`;
    this.updateThemeDisplay({ name: this.levelName });
    this.generateGrid();
    this.updateGridSizeVariable();
    this.buildGrid();
    this.render();

    if (this.currentLevel === 1 && !this.demoWordShown) {
      this.autoFindDemoWord();
      this.demoWordShown = true;
    }

    document.getElementById("levelCompleteMessage").style.display = "none";
  }

  rebuildPlacements() {
    this.placements.clear();
    for (const [word, path] of this.wordPaths.entries()) {
      for (const [r, c] of path) {
        const key = `${r},${c}`;
        if (!this.placements.has(key)) this.placements.set(key, new Set());
        this.placements.get(key).add(word);
      }
      if (path.length > 0) {
        this.wordStartCells.set(word, { row: path[0][0], col: path[0][1] });
      }
    }
  }

  autoFindDemoWord() {
    if (this.words.length === 0) return;
    const randomIndex = Math.floor(Math.random() * this.words.length);
    const demoWord = this.words[randomIndex];
    this.foundWords.add(demoWord);
    this.showMessage(`Слово "${demoWord}" уже найдено — попробуйте найти остальные!`, "level-complete");
    this.render();
  }

  updateThemeDisplay(theme) {
    document.getElementById("currentTheme").textContent = theme.name;
    document.getElementById("levelProgress").textContent = `Уровень: ${this.currentLevel}`;
  }

  generateGrid() {
    this.grid = Array.from({ length: this.gridSize }, () =>
      Array(this.gridSize).fill(null)
    );
    this.wordPaths.clear();
    this.wordStartCells.clear();

    for (const word of this.words) {
      this.placeWordSnaking(word);
    }
    this.placeExtraWords();
    this.fillEmptyCells();
  }

  placeWordSnaking(word, maxAttempts = 200) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const startRow = Math.floor(Math.random() * this.gridSize);
      const startCol = Math.floor(Math.random() * this.gridSize);

      if (this.grid[startRow][startCol] !== null) continue;

      const path = this.buildSnakingPath(word, startRow, startCol);
      if (path && path.length === word.length) {
        this.commitSnakingPath(word, path);
        return true;
      }
    }
    return false;
  }

  buildSnakingPath(word, startRow, startCol) {
    const path = [[startRow, startCol]];
    const visited = new Set([`${startRow},${startCol}`]);

    const dfs = (index, prevDir) => {
      if (index === word.length) return true;

      const [r, c] = path[index - 1];
      const neighbors = this.biasedNeighbors(r, c, prevDir);

      for (const [nr, nc] of neighbors) {
        const key = `${nr},${nc}`;
        if (visited.has(key)) continue;
        if (this.grid[nr][nc] !== null) continue;

        path.push([nr, nc]);
        visited.add(key);

        if (dfs(index + 1, [nr - r, nc - c])) return true;

        path.pop();
        visited.delete(key);
      }
      return false;
    };

    if (dfs(1, null)) return path;
    return null;
  }

  biasedNeighbors(r, c, prevDir) {
    const dirs = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1], [0, 1],
      [1, -1], [1, 0], [1, 1],
    ];

    const candidates = [];
    for (const [dr, dc] of dirs) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < this.gridSize && nc >= 0 && nc < this.gridSize) {
        candidates.push([nr, nc, dr, dc]);
      }
    }

    if (!prevDir) {
      for (let i = candidates.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
      }
      return candidates.map(([nr, nc]) => [nr, nc]);
    }

    const [pdr, pdc] = prevDir;
    const dot = ([dr, dc]) => dr * pdr + dc * pdc;
    const tier = ([, , dr, dc]) => {
      const d = dot([dr, dc]);
      if (d > 0.7) return 0;
      if (d > 0.1) return 1;
      if (d > -0.4) return 2;
      return 3;
    };

    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }

    candidates.sort((a, b) => tier(a) - tier(b));
    return candidates.map(([nr, nc]) => [nr, nc]);
  }

  commitSnakingPath(word, path) {
    this.wordPaths.set(word, path);
    this.wordStartCells.set(word, { row: path[0][0], col: path[0][1] });

    for (let i = 0; i < path.length; i++) {
      const [r, c] = path[i];
      this.grid[r][c] = word[i];

      const key = `${r},${c}`;
      if (!this.placements.has(key)) this.placements.set(key, new Set());
      this.placements.get(key).add(word);
    }
  }

  placeExtraWords() {
    const candidates = allWords
      .filter(
        (word) =>
          !word.includes(this.themeLetter) &&
          !this.words.includes(word) &&
          word.length <= this.gridSize,
      )
      .sort(() => Math.random() - 0.5)
      .slice(0, 80);

    for (const word of candidates) {
      if (this.placeWordSnaking(word, 50)) {
        this.extraWords.push(word);
      }
    }
  }

  fillEmptyCells() {
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        if (this.grid[i][j] === null) {
          this.grid[i][j] = RUSSIAN_ALPHABET[Math.floor(Math.random() * RUSSIAN_ALPHABET.length)];
        }
      }
    }
  }

  startSelection(row, col) {
    if (this.isCellFound(row, col)) return;

    if (this.hintCells.size > 0) {
      this.hintCells.clear();
      this.updateCellStates();
    }

    this.isSelecting = true;
    this.selectedCells = [];
    this.addToSelection(row, col);
  }

  isCellFound(row, col) {
    const cellKey = `${row},${col}`;
    const wordSet = this.placements.get(cellKey);
    if (!wordSet) return false;
    for (let word of wordSet) {
      if (this.foundWords.has(word)) return true;
    }
    return false;
  }

  addToSelection(row, col) {
    if (!this.isSelecting) return;
    if (this.isCellFound(row, col)) return;

    const existingIdx = this.selectedCells.findIndex(([r, c]) => r === row && c === col);
    if (existingIdx !== -1) {
      this.selectedCells = this.selectedCells.slice(0, existingIdx + 1);
      this.renderSelectionHighlight();
      return;
    }

    if (this.selectedCells.length > 0) {
      const [lastR, lastC] = this.selectedCells[this.selectedCells.length - 1];
      if (Math.abs(row - lastR) > 1 || Math.abs(col - lastC) > 1) return;
    }

    this.selectedCells.push([row, col]);
    this.renderSelectionHighlight();
  }

  renderSelectionHighlight() {
    document.querySelectorAll(".grid-cell.selected").forEach((el) => el.classList.remove("selected"));
    const gridEl = document.getElementById("grid");
    for (const [r, c] of this.selectedCells) {
      const index = r * this.gridSize + c;
      if (gridEl.children[index]) {
        gridEl.children[index].classList.add("selected");
      }
    }
  }

  stopSelection() {
    if (this.isSelecting && this.selectedCells.length > 0) {
      this.checkSelectedWord();
    }
    this.isSelecting = false;
  }

  checkSelectedWord() {
    if (this.selectedCells.length < 2) {
      this.selectedCells = [];
      this.render();
      return;
    }

    const selectedWord = this.selectedCells.map(([r, c]) => this.grid[r][c]).join("");
    const reversedWord = selectedWord.split("").reverse().join("");

    const foundWord = this.words.find(word => !this.foundWords.has(word) && (word === selectedWord || word === reversedWord));

    if (foundWord) {
      this.foundWords.add(foundWord);
      this.saveProgress();
      this.showMessage(`Найдено: ${foundWord}!`, "success");
      if (this.foundWords.size === this.words.length) {
        this.levelComplete();
      }
    } else {
      const wordInAll = allWords.find((w) => w === selectedWord || w === reversedWord);
      if (wordInAll && !this.words.includes(wordInAll) && !this.foundExtraWords.has(wordInAll)) {
        this.foundExtraWords.add(wordInAll);
        this.extraWordsFoundCount++;
        if (this.extraWordsFoundCount % 3 === 0) {
          this.hintsUsed++;
          this.giveHint();
          this.saveProgress();
        } else {
          this.saveProgress();
          this.showMessage("Слово из другого уровня!", "success");
        }
      } else {
        this.showMessage("Данного слова нет в текущем словаре", "error");
      }
    }

    this.selectedCells = [];
    this.render();
  }

  giveHint() {
    const unfound = this.words.filter((word) => !this.foundWords.has(word));
    if (unfound.length === 0) return;
    const hintWord = unfound[Math.floor(Math.random() * unfound.length)];

    const path = this.wordPaths.get(hintWord);
    if (!path || path.length === 0) return;

    path.forEach(([r, c]) => this.hintCells.add(`${r},${c}`));
    this.showMessage(`Подсказка: одно из слов подсвечено на поле!`, "level-complete");
  }

  levelComplete() {
    let message = "";
    if (this.hintsUsed === 0) {
      message = "Без подсказок. Идеально ✨";
    } else if (this.hintsUsed === 1) {
      message = "Так держать! Всего одна подсказка";
    } else if (this.hintsUsed === 2) {
      message = "Две подсказки? Неплохо :)";
    } else {
      message = "Сложный уровень? Не сдавайся!";
    }

    const confettiDiv = document.querySelector("#levelCompleteMessage .confetti");
    confettiDiv.textContent = message;

    document.getElementById("levelCompleteMessage").style.display = "block";
    this.saveProgress();
  }

  showMessage(text, type) {
    const messageEl = document.getElementById("message");
    clearTimeout(this._msgTimer);
    messageEl.textContent = text;
    messageEl.className = `message ${type}`;
    messageEl.style.display = "block";
    this._msgTimer = setTimeout(() => { messageEl.style.display = "none"; }, 2000);
  }

  buildGrid() {
    const gridEl = document.getElementById("grid");
    const wrapper = gridEl.parentElement;

    wrapper.querySelector(".word-lines-svg")?.remove();

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("word-lines-svg");
    wrapper.appendChild(svg);

    gridEl.innerHTML = "";
    gridEl.style.gridTemplateColumns = `repeat(${this.gridSize}, 1fr)`;

    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        const cell = document.createElement("div");
        cell.className = "grid-cell";
        cell.dataset.row = i;
        cell.dataset.col = j;
        cell.textContent = this.grid[i][j];
        gridEl.appendChild(cell);
      }
    }

    gridEl.onpointerdown = (e) => {
      const cell = e.target.closest(".grid-cell");
      if (!cell) return;
      gridEl.setPointerCapture(e.pointerId);
      this.startSelection(parseInt(cell.dataset.row), parseInt(cell.dataset.col));
    };

    this.renderWordList();
  }

  updateCellStates() {
    const gridEl = document.getElementById("grid");
    const cells = gridEl.querySelectorAll(".grid-cell");

    cells.forEach(cell => {
      const r = parseInt(cell.dataset.row);
      const c = parseInt(cell.dataset.col);
      const cellKey = `${r},${c}`;

      const isSelected = this.selectedCells.some(([sr, sc]) => sr === r && sc === c);
      const isFound = this.isCellFound(r, c);
      const isHint = this.hintCells.has(cellKey);

      let className = "grid-cell";
      if (isFound) className += " found";
      if (isSelected) className += " selected";
      if (isHint) className += " hint";
      if (!isFound && !isSelected && !isHint && this.grid[r][c] === this.themeLetter) {
        className += " theme-letter";
      }
      cell.className = className;
    });
  }

  drawLines() {
    const gridEl = document.getElementById("grid");
    const wrapper = gridEl.parentElement;
    const svg = wrapper.querySelector(".word-lines-svg");
    if (!svg) return;

    svg.innerHTML = "";

    const wrapperRect = wrapper.getBoundingClientRect();
    const cells = gridEl.querySelectorAll(".grid-cell");
    if (cells.length === 0) return;

    const sampleRect = cells[0].getBoundingClientRect();
    const strokeWidth = sampleRect.width * 0.38;

    let colorIdx = 0;
    for (const word of this.words) {
      if (!this.foundWords.has(word)) continue;
      const path = this.wordPaths.get(word);
      if (!path || path.length < 2) continue;

      const pts = [];
      for (const [r, c] of path) {
        const cellEl = cells[r * this.gridSize + c];
        if (!cellEl) continue;
        const rect = cellEl.getBoundingClientRect();
        pts.push({
          x: rect.left - wrapperRect.left + rect.width / 2,
          y: rect.top - wrapperRect.top + rect.height / 2,
        });
      }

      if (pts.length < 2) continue;

      const waypoints = [pts[0]];
      for (let i = 1; i < pts.length - 1; i++) {
        const prev = waypoints[waypoints.length - 1];
        const curr = pts[i];
        const next = pts[i + 1];
        const cross = Math.abs(
          (curr.x - prev.x) * (next.y - prev.y) - (curr.y - prev.y) * (next.x - prev.x)
        );
        if (cross > 1) waypoints.push(curr);
      }
      waypoints.push(pts[pts.length - 1]);

      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("points", waypoints.map(p => `${p.x},${p.y}`).join(" "));
      polyline.setAttribute("stroke", LINE_COLORS[colorIdx % LINE_COLORS.length]);
      polyline.setAttribute("stroke-width", strokeWidth);
      polyline.setAttribute("stroke-linecap", "round");
      polyline.setAttribute("stroke-linejoin", "round");
      polyline.setAttribute("fill", "none");
      svg.appendChild(polyline);

      colorIdx++;
    }
  }

  render() {
    this.updateCellStates();
    this.renderFoundWords();
    this.renderWordList();
    this.updateProgress();
    this.drawLines();
  }

  renderWordList() {
    const section = document.getElementById("wordListSection");
    if (!section) return;
    if (this.abGroup !== "B") {
      section.style.display = "none";
      return;
    }
    section.style.display = "block";
    document.getElementById("wordList").innerHTML = this.words
      .map(word => `<span class="word-list-item${this.foundWords.has(word) ? " found" : ""}">${word}</span>`)
      .join("");
  }

  renderFoundWords() {
    const container = document.getElementById("foundWords");
    if (this.abGroup === "B") {
      container.innerHTML = "";
      return;
    }
    const foundArray = Array.from(this.foundWords);
    if (foundArray.length === 0) {
      container.innerHTML = '<div class="empty-words">Пока ничего не найдено</div>';
      return;
    }
    container.innerHTML = foundArray
      .map((word) => `<span class="found-word-badge">${word}</span>`)
      .join("");
  }

  updateProgress() {
    const found = this.foundWords.size;
    const total = this.words.length;
    const percentage = (found / total) * 100;

    document.getElementById("foundCount").textContent = `${found}/${total}`;
    document.getElementById("progressFill").style.width = `${percentage}%`;

    const progressInCycle = this.extraWordsFoundCount % 3;
    const hintBadge = document.getElementById("hintProgressBadge");
    if (hintBadge) {
      hintBadge.dataset.progress = progressInCycle;
      hintBadge.querySelectorAll(".hint-pip").forEach((pip, idx) => {
        pip.classList.toggle("filled", idx < progressInCycle);
      });
    }
  }
}

const game = new WordSearchGame();
