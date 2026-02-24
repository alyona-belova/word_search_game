const themes = [
  {
    name: "Кинопроизводство",
    words: [
      "0KHQptCV0J3QkNCg0JjQmQ==",
      "0JrQkNCh0KLQmNCd0JM=",
      "0J/QoNCe0JTQrtCh0JXQoA==",
      "0JTQldCa0J7QoNCQ0KbQmNCv",
      "0JzQntCd0KLQkNCW",
      "0JTQo9CR0JvQrA==",
      "0KHQqtCB0JzQmtCQ",
      "0KDQldCW0JjQodCh0JXQoA==",
    ],
  },
  {
    name: "Космос",
    words: [
      "0J7QoNCR0JjQotCQ",
      "0KHQotCr0JrQntCS0JrQkA==",
      "0JrQkNCf0KHQo9Cb0JA=",
      "0K3QmtCY0J/QkNCW",
      "0JzQntCU0KPQm9Cs",
      "0KHQn9Cj0KLQndCY0Jo=",
      "0KDQkNCa0JXQotCQ",
      "0KLQoNCQ0JXQmtCi0J7QoNCY0K8=",
    ],
  },
  {
    name: "Городская среда",
    words: [
      "0JHQo9Cb0KzQktCQ0KA=",
      "0KTQntCd0JDQoNCs",
      "0KLQoNCe0KLQo9CQ0KA=",
      "0KHQmtCS0JXQoA==",
      "0JLQmNCi0KDQmNCd0JA=",
      "0J/QldCg0JXQpdCe0JQ=",
      "0J7QodCi0JDQndCe0JLQmtCQ",
      "0JDQoNCa0JA=",
    ],
  },
  {
    name: "Машинное обучение",
    words: [
      "0JTQkNCi0JDQodCV0KI=",
      "0J/QoNCY0JfQndCQ0Jo=",
      "0JzQldCi0KDQmNCa0JA=",
      "0JzQntCU0JXQm9Cs",
      "0JPQoNCQ0JTQmNCV0J3Qog==",
      "0JrQm9CQ0KHQotCV0KA=",
      "0KDQldCT0KDQldCh0KHQmNCv",
      "0J/QoNCV0JTQodCa0JDQl9CQ0J3QmNCV",
    ],
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
    this.demoWordShown = false;

    this.init();
  }

  init() {
    this.loadLevel(this.currentLevelIndex);
    this.setupEventListeners();
  }

  setupEventListeners() {
    document.addEventListener("mouseup", () => this.stopSelection());
    document
      .getElementById("nextLevelBtn")
      .addEventListener("click", () => this.nextLevel());
  }

  decodeWord(encoded) {
    return decodeURIComponent(escape(atob(encoded)));
  }

  loadLevel(index) {
    if (index >= themes.length) {
      this.showGameComplete();
      return;
    }

    const theme = themes[index];
    this.words = [...theme.words]
      .map((w) => this.decodeWord(w).toUpperCase())
      .sort((a, b) => b.length - a.length);

    this.foundWords.clear();
    this.selectedCells.clear();
    this.placements.clear();

    this.updateThemeDisplay(theme);
    this.generateGrid();
    this.render();

    if (index === 0 && !this.demoWordShown) {
      this.autoFindDemoWord();
      this.demoWordShown = true;
    }

    document.getElementById("levelCompleteMessage").style.display = "none";
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
    document.getElementById("levelProgress").textContent = `Уровень ${
      this.currentLevelIndex + 1
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
      "Я",
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
    this.showMessage("Поздравляем! Все слова найдены 🎉", "level-complete");
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
