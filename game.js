const themes = [
  {
    name: "Артефакты",
    words: ["АМУЛЕТ", "ПОСОХ", "ГРИМУАР", "КРИСТАЛЛ", "СВИТОК", "ТАЛИСМАН", "ПЕРСТЕНЬ"],
  },
  {
    name: "Магические ритуалы",
    words: ["ЗАКЛИНАНИЕ", "ОБРЯД", "ПРИВОРОТ", "ЛЕВИТАЦИЯ", "ПРИЗЫВ", "МЕДИТАЦИЯ", "ПОСВЯЩЕНИЕ"],
  },
  {
    name: "Ингредиенты для зелья",
    words: ["МАНДРАГОРА", "ЛАВАНДА", "ПОЛЫНЬ", "СЕРА", "РТУТЬ", "ЗОЛА", "РОСА"],
  },
  {
    name: "Мифические существа",
    words: ["ФЕНИКС", "ВАСИЛИСК", "СФИНКС", "ГОБЛИН", "ЕДИНОРОГ", "СИРЕНА", "ГРИФОН"],
  },
];

class WordSearchGame {
  constructor() {
    this.currentLevelIndex = 0;
    this.grid = [];
    this.words = [];
    this.foundWords = new Set();
    this.selectedCells = new Set();
    this.isSelecting = false;
    this.gridSize = 12;
    this.placements = new Map();

    this.init();
  }

  init() {
    this.loadLevel(this.currentLevelIndex);
    this.setupEventListeners();
  }

  updateGridSizeVariable() {
    document.documentElement.style.setProperty('--grid-size', this.gridSize);
  }

  setupEventListeners() {
    document.addEventListener("mouseup", () => this.stopSelection());
    document
      .getElementById("nextLevelBtn")
      .addEventListener("click", () => this.nextLevel());

    document.getElementById('grid').addEventListener('touchmove', (e) => {
      e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchstart', (e) => {
      if (e.target.classList.contains('grid-cell')) {
        e.preventDefault();
      }
    }, { passive: false });
  }

  loadLevel(index) {
    if (index >= themes.length) {
      this.showGameComplete();
      return;
    }

    const theme = themes[index];
    this.words = [...theme.words]
      .map((w) => w.toUpperCase())
      .sort((a, b) => b.length - a.length); // longest first

    this.foundWords.clear();
    this.selectedCells.clear();
    this.placements.clear();

    this.updateThemeDisplay(theme);
    this.generateGrid();
    this.updateGridSizeVariable();
    this.render();

    document.getElementById("levelCompleteMessage").style.display = "none";
  }

  updateThemeDisplay(theme) {
    document.getElementById("currentTheme").textContent = theme.name;
    document.getElementById("levelProgress").textContent = `Уровень ${this.currentLevelIndex + 1
      }/${themes.length}`;
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
        return false; // no overlapping allowed
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
      "Я"
    ];

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
      this.render();
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
  }

  nextLevel() {
    this.currentLevelIndex++;
    if (this.currentLevelIndex < themes.length) {
      this.loadLevel(this.currentLevelIndex);
    } else {
      this.showGameComplete();
    }
  }

  showGameComplete() {
    this.showMessage(
      "Поздравляем! Все слова восстановлены, а заклинания снова активны. Великий Архив вновь сияет магическим знанием, и теперь вы можете считать себя полноценным хранителем тайн Академии.",
      "level-complete",
    );
    document.getElementById("nextLevelBtn").style.display = "none";
  }

  showMessage(text, type) {
    const messageEl = document.getElementById("message");
    messageEl.textContent = text;
    messageEl.className = `message ${type}`;
    messageEl.style.display = "block";

    setTimeout(() => {
      if (type !== "level-complete") {
        messageEl.style.display = "none";
      }
    }, 2000);
  }

  handleTouchStart(event) {
    event.preventDefault();
    const cell = event.target;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);
    this.startSelection(row, col);
  }

  handleTouchMove(event) {
    event.preventDefault();
    const touch = event.touches[0];
    const element = document.elementFromPoint(touch.clientX, touch.clientY);

    if (element && element.classList.contains('grid-cell')) {
      const row = parseInt(element.dataset.row);
      const col = parseInt(element.dataset.col);
      this.addToSelection(row, col);
    }
  }

  handleTouchEnd(event) {
    event.preventDefault();
    this.stopSelection();
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
        data-row="${i}" data-col="${j}"
        ontouchstart="game.handleTouchStart(event)"
        ontouchmove="game.handleTouchMove(event)"
        ontouchend="game.handleTouchEnd(event)"
        ontouchcancel="game.handleTouchEnd(event)"
        onmousedown="game.startSelection(${i}, ${j})"
        onmouseover="game.addToSelection(${i}, ${j})"
        onmouseup="game.stopSelection()"
        ondblclick="game.selectedCells.clear(); game.render();"
      >${this.grid[i][j]}</div>`;
      }
    }

    gridEl.innerHTML = html;
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
