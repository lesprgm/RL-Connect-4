const modeSelect = document.getElementById("mode-select");
const newGameButton = document.getElementById("new-game-button");
const modeDescription = document.getElementById("mode-description");
const statusText = document.getElementById("status-text");
const modeText = document.getElementById("mode-text");
const sessionText = document.getElementById("session-text");
const boardEl = document.getElementById("board");
const columnButtonsEl = document.getElementById("column-buttons");

let modes = [];
let gameState = null;

function playerLabel(player) {
  if (player === 1) return "Yellow";
  if (player === 2) return "Red";
  return "None";
}

function describeStatus(state) {
  if (!state) return "Choose a mode to begin.";
  if (state.winner) return `${playerLabel(state.winner)} wins.`;
  if (state.is_draw) return "The game is a draw.";
  if (state.mode === "human_vs_human") {
    return `${playerLabel(state.current_player)} to move.`;
  }
  return state.current_player === 1 ? "Your turn." : "Agent is thinking...";
}

function renderBoard(state) {
  boardEl.innerHTML = "";
  columnButtonsEl.innerHTML = "";

  for (let column = 0; column < 7; column += 1) {
    const button = document.createElement("button");
    button.className = "column-button";
    button.textContent = `Drop ${column + 1}`;
    const disabled =
      !state ||
      state.is_over ||
      !state.available_columns.includes(column) ||
      (state.mode !== "human_vs_human" && state.current_player !== 1);
    button.disabled = disabled;
    button.addEventListener("click", () => makeMove(column));
    columnButtonsEl.appendChild(button);
  }

  const board = state?.board ?? Array.from({ length: 6 }, () => Array(7).fill(0));
  board.forEach((row) => {
    row.forEach((cell) => {
      const slot = document.createElement("div");
      slot.className = "cell";
      if (cell === 1) slot.classList.add("player-1");
      if (cell === 2) slot.classList.add("player-2");
      boardEl.appendChild(slot);
    });
  });
}

function renderState(state) {
  gameState = state;
  statusText.textContent = describeStatus(state);
  modeText.textContent = state ? state.active_mode_label : "Not started";
  sessionText.textContent = state ? state.session_id.slice(0, 8) : "Not started";
  renderBoard(state);
}

async function fetchModes() {
  const response = await fetch("/api/modes");
  const data = await response.json();
  modes = data.modes;
  modeSelect.innerHTML = "";
  modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode.id;
    option.textContent = mode.label;
    modeSelect.appendChild(option);
  });
  updateModeDescription();
}

function updateModeDescription() {
  const selected = modes.find((mode) => mode.id === modeSelect.value);
  modeDescription.textContent = selected ? selected.description : "";
}

async function startNewGame() {
  const response = await fetch("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: modeSelect.value }),
  });
  const data = await response.json();
  renderState(data);
}

async function makeMove(column) {
  if (!gameState) return;
  const response = await fetch(`/api/games/${gameState.session_id}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column }),
  });
  const data = await response.json();
  if (!response.ok) {
    statusText.textContent = data.error;
    return;
  }
  renderState(data);
}

modeSelect.addEventListener("change", updateModeDescription);
newGameButton.addEventListener("click", startNewGame);

fetchModes().then(() => renderBoard(null));
