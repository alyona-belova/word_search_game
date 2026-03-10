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
  "А",
  "Б",
  "В",
  "Г",
  "Д",
  "Е",
  "Ё",
  "Ж",
  "З",
  "И",
  "Й",
  "К",
  "Л",
  "М",
  "Н",
  "О",
  "П",
  "Р",
  "С",
  "Т",
  "У",
  "Ф",
  "Х",
  "Ц",
  "Ч",
  "Ш",
  "Щ",
  "Ъ",
  "Ы",
  "Ь",
  "Э",
  "Ю",
  "Я",
];

class WordSearchGame {
  constructor() {
    this.currentLevel = 1;
    this.grid = [];
    this.words = [];
    this.foundWords = new Set();
    this.selectedCells = new Set();
    this.isSelecting = false;
    this.gridSize = 12;
    this.placements = new Map();
    this.demoWordShown = false;
    this.levelName = "";

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
        this.updateThemeDisplay({ name: this.levelName });
        this.rebuildPlacements();
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
    document.addEventListener("mouseup", () => this.stopSelection());
    document.addEventListener("touchend", () => this.stopSelection());
    document
      .getElementById("nextLevelBtn")
      .addEventListener("click", () => this.nextLevel());
  }

  loadLevel() {
    const randomLetter =
      RUSSIAN_ALPHABET[Math.floor(Math.random() * RUSSIAN_ALPHABET.length)];
    const wordsWithLetter = allWords.filter(
      (word) => word.includes(randomLetter) && word.length <= this.gridSize - 1,
    );

    if (wordsWithLetter.length < 8) {
      return this.loadLevel(Math.random());
    }

    const short = wordsWithLetter.filter((w) => w.length >= 4 && w.length <= 5);
    const medium = wordsWithLetter.filter(
      (w) => w.length >= 6 && w.length <= 7,
    );
    const long = wordsWithLetter.filter((w) => w.length >= 8);

    function pickRandom(arr, count) {
      return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
    }

    this.words = [
      ...pickRandom(short, 2),
      ...pickRandom(medium, 3),
      ...pickRandom(long, 3),
    ].sort((a, b) => b.length - a.length);

    this.foundWords.clear();
    this.selectedCells.clear();
    this.placements.clear();

    this.levelName = `Буква "${randomLetter}"`;
    this.updateThemeDisplay({ name: this.levelName });
    this.generateGrid();
    this.updateGridSizeVariable();
    this.render();

    if (this.currentLevel === 1 && !this.demoWordShown) {
      this.autoFindDemoWord();
      this.demoWordShown = true;
    }

    document.getElementById("levelCompleteMessage").style.display = "none";
  }

  rebuildPlacements() {
    this.placements.clear();
    for (let word of this.words) {
      for (let i = 0; i < this.gridSize; i++) {
        for (let j = 0; j < this.gridSize; j++) {
          if (this.grid[i][j] === word[0]) {
            const right = this.checkWordAt(word, i, j, 0, 1);
            const down = this.checkWordAt(word, i, j, 1, 0);
            if (right)
              this.commitPlacement(word, { row: i, col: j, direction: [0, 1] });
            if (down)
              this.commitPlacement(word, { row: i, col: j, direction: [1, 0] });
          }
        }
      }
    }
  }

  checkWordAt(word, row, col, dr, dc) {
    for (let i = 0; i < word.length; i++) {
      const r = row + dr * i;
      const c = col + dc * i;

      if (r >= this.gridSize || c >= this.gridSize) return false;
      if (this.grid[r][c] !== word[i]) return false;
    }
    return true;
  }

  autoFindDemoWord() {
    if (this.words.length === 0) return;
    const randomIndex = Math.floor(Math.random() * this.words.length);
    const demoWord = this.words[randomIndex];
    this.foundWords.add(demoWord);

    this.showMessage(
      `Слово "${demoWord}" уже найдено — попробуйте найти остальные!`,
      "level-complete",
    );

    this.render();
  }

  updateThemeDisplay(theme) {
    document.getElementById("currentTheme").textContent = theme.name;
    document.getElementById("levelProgress").textContent =
      `Уровень: ${this.currentLevel}`;
  }

  generateGrid() {
    this.grid = Array.from({ length: this.gridSize }, () =>
      Array(this.gridSize).fill(null),
    );

    for (const word of this.words) {
      this.placeWord(word);
    }

    this.fillEmptyCells();
  }

  placeWord(word) {
    const directions = [
      [0, 1], // right
      [1, 0], // down
    ];

    const validPlacements = [];

    for (const direction of directions) {
      for (let row = 0; row < this.gridSize; row++) {
        for (let col = 0; col < this.gridSize; col++) {
          if (this.canPlaceWord(word, row, col, direction)) {
            validPlacements.push({ row, col, direction });
          }
        }
      }
    }

    if (validPlacements.length === 0) {
      throw new Error(`Cannot place word: ${word}`);
    }

    const choice =
      validPlacements[Math.floor(Math.random() * validPlacements.length)];

    this.commitPlacement(word, choice);
  }

  canPlaceWord(word, startRow, startCol, direction) {
    for (let i = 0; i < word.length; i++) {
      const row = startRow + direction[0] * i;
      const col = startCol + direction[1] * i;

      if (row < 0 || row >= this.gridSize || col < 0 || col >= this.gridSize) {
        return false;
      }

      if (this.grid[row][col] !== null) {
        return false; // no overlap
      }
    }

    return true;
  }

  commitPlacement(word, { row, col, direction }) {
    for (let i = 0; i < word.length; i++) {
      const r = row + direction[0] * i;
      const c = col + direction[1] * i;

      this.grid[r][c] = word[i];

      const key = `${r},${c}`;
      if (!this.placements.has(key)) {
        this.placements.set(key, new Set());
      }
      this.placements.get(key).add(word);
    }
  }

  fillEmptyCells() {
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        if (this.grid[i][j] === null) {
          this.grid[i][j] =
            RUSSIAN_ALPHABET[
            Math.floor(Math.random() * RUSSIAN_ALPHABET.length)
            ];
        }
      }
    }
  }

  startSelection(row, col) {
    if (this.isCellFound(row, col)) return;

    this.isSelecting = true;
    this.selectedCells.clear();
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

    const key = `${row},${col}`;
    if (!this.selectedCells.has(key)) {
      this.selectedCells.add(key);
      const gridEl = document.getElementById("grid");
      const index = row * this.gridSize + col;
      gridEl.children[index].classList.add("selected");
    }
  }

  stopSelection() {
    if (this.isSelecting && this.selectedCells.size > 0) {
      this.checkSelectedWord();
    }
    this.isSelecting = false;
  }

  checkSelectedWord() {
    const cells = Array.from(this.selectedCells)
      .map((cell) => cell.split(",").map(Number))
      .sort((a, b) => {
        if (a[0] !== b[0]) return a[0] - b[0];
        return a[1] - b[1];
      });

    if (cells.length < 2) {
      this.selectedCells.clear();
      this.render();
      return;
    }

    if (!this.checkStraightLine(cells)) {
      this.showMessage("Выбор должен осуществляться по прямой линии", "error");
      this.selectedCells.clear();
      this.render();
      return;
    }

    const selectedWord = cells
      .map(([row, col]) => this.grid[row][col])
      .join("");

    const reversedWord = selectedWord.split("").reverse().join("");

    const foundWord = this.words.find(
      (word) =>
        !this.foundWords.has(word) &&
        (word === selectedWord || word === reversedWord),
    );

    if (foundWord) {
      this.foundWords.add(foundWord);
      this.saveProgress();

      this.showMessage(`Найдено: ${foundWord}!`, "success");

      if (this.foundWords.size === this.words.length) {
        this.levelComplete();
      }
    } else {
      this.showMessage("Данного слова нет в текущей теме", "error");
    }

    this.selectedCells.clear();
    this.render();
  }

  checkStraightLine(cells) {
    if (cells.length < 2) return true;

    const [first, second] = cells;
    const rowDiff = second[0] - first[0];
    const colDiff = second[1] - first[1];

    if (Math.abs(rowDiff) > 1 || Math.abs(colDiff) > 1) return false;
    if (rowDiff === 0 && colDiff === 0) return false;

    for (let i = 2; i < cells.length; i++) {
      const expectedRow = first[0] + rowDiff * i;
      const expectedCol = first[1] + colDiff * i;

      if (cells[i][0] !== expectedRow || cells[i][1] !== expectedCol) {
        return false;
      }
    }

    return true;
  }

  levelComplete() {
    this.showMessage("Ура! Уровень пройден!", "level-complete");
    document.getElementById("levelCompleteMessage").style.display = "block";

    this.currentLevel++;
    this.saveProgress();
  }

  nextLevel() {
    this.saveProgress();
    this.loadLevel();
  }

  showMessage(text, type) {
    const messageEl = document.getElementById("message");
    messageEl.textContent = text;
    messageEl.className = `message ${type}`;
    messageEl.style.display = "block";

    setTimeout(() => {
      messageEl.style.display = "none";
    }, 2000);
  }

  render() {
    this.renderGrid();
    this.renderFoundWords();
    this.updateProgress();
  }

  renderGrid() {
    const gridEl = document.getElementById("grid");
    gridEl.style.gridTemplateColumns = `repeat(${this.gridSize}, 1fr)`;

    let html = "";
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        const cellKey = `${i},${j}`;
        const isSelected = this.selectedCells.has(cellKey);
        const isFound = this.isCellFound(i, j);

        let cellClass = "grid-cell";
        if (isFound) cellClass += " found";
        if (isSelected) cellClass += " selected";

        html += `<div class="${cellClass}"
        onmousedown="game.startSelection(${i}, ${j})"
        onmouseover="game.addToSelection(${i}, ${j})"
        ontouchstart="game.startSelection(${i}, ${j})"
        ontouchmove="game.handleTouchMove(event)"
        ondblclick="game.selectedCells.clear(); game.render();"
        >${this.grid[i][j]}</div>`;
      }
    }

    gridEl.innerHTML = html;
  }

  handleTouchMove(event) {
    event.preventDefault();

    const touch = event.touches[0];
    const element = document.elementFromPoint(touch.clientX, touch.clientY);

    if (!element) return;

    const cell = element.closest(".grid-cell");
    if (!cell) return;

    const index = Array.from(cell.parentNode.children).indexOf(cell);

    const row = Math.floor(index / this.gridSize);
    const col = index % this.gridSize;

    this.addToSelection(row, col);
  }

  renderFoundWords() {
    const container = document.getElementById("foundWords");
    const foundArray = Array.from(this.foundWords);

    if (foundArray.length === 0) {
      container.innerHTML = "";
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
  }
}

const game = new WordSearchGame();
