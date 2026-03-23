declare function ym(
  counterId: number,
  action: string,
  ...args: unknown[]
): void;

const METRICA_COUNTER_ID = 107726491;

function track(goal: string, params?: Record<string, unknown>): void {
  if (typeof ym === "undefined") return;
  ym(METRICA_COUNTER_ID, "reachGoal", goal, params ?? {});
}
function setVisitParams(params: Record<string, unknown>): void {
  if (typeof ym === "undefined") return;
  const nested: Record<string, Record<string, number>> = {};
  for (const [k, v] of Object.entries(params)) {
    nested[k] = { [String(v)]: 1 };
  }
  ym(METRICA_COUNTER_ID, "params", nested);
}

const _sessionStart = Date.now();
let _levelStart = 0;

function trackSessionStart(abGroup: string): void {
  const isReturning = Boolean(localStorage.getItem("tutorialSeen"));

  const visitCount = parseInt(localStorage.getItem("visitCount") ?? "0") + 1;
  localStorage.setItem("visitCount", String(visitCount));

  const hourOfDay = new Date().getHours();

  track("session_start", {
    is_returning: isReturning,
    ab_group: abGroup,
    visit_count: visitCount,
    hour_of_day: hourOfDay,
  });
  setVisitParams({
    ab_group: abGroup,
    is_returning: isReturning ? 1 : 0,
    visit_count: visitCount,
    hour_of_day: hourOfDay,
  });

  window.addEventListener("beforeunload", () => {
    const duration = Math.round((Date.now() - _sessionStart) / 1000);
    track("session_end", { duration_sec: duration });
  });
}

function resetLevelTimer(): void {
  _levelStart = Date.now();
}

function trackLevelStart(
  level: number,
  themeLetter: string,
  levelSeq: number,
): void {
  _levelStart = Date.now();
  track("level_started", {
    level,
    theme_letter: themeLetter,
    level_seq: levelSeq,
  });
  setVisitParams({ level, level_status: "in_progress", level_seq: levelSeq });
}

function trackFirstWordFound(level: number): void {
  const timeSec =
    _levelStart > 0 ? Math.round((Date.now() - _levelStart) / 1000) : 0;
  track("first_word_found", { level, time_to_first_word_sec: timeSec });
  setVisitParams({ time_to_first_word_sec: timeSec });
}

function trackLevelComplete(
  level: number,
  hintsUsed: number,
  wordsTotal: number,
): void {
  track("level_completed", {
    level,
    hints_used: hintsUsed,
    words_total: wordsTotal,
  });
  setVisitParams({ level_status: "completed", hints_used: hintsUsed });
}

function trackHintReceived(
  level: number,
  wordsFound: number,
  wordsTotal: number,
): void {
  const completion_pct =
    wordsTotal > 0 ? Math.round((wordsFound / wordsTotal) * 100) : 0;
  track("hint_received", {
    level,
    words_found: wordsFound,
    words_total: wordsTotal,
    completion_pct,
  });
}

function trackDropOff(
  level: number,
  wordsFound: number,
  wordsTotal: number,
  hintsUsed: number,
): void {
  const completion_pct =
    wordsTotal > 0 ? Math.round((wordsFound / wordsTotal) * 100) : 0;
  track("drop_off", {
    level,
    words_found: wordsFound,
    words_total: wordsTotal,
    completion_pct,
  });
  setVisitParams({
    level_status: "dropped",
    drop_off_pct: completion_pct,
    hints_used: hintsUsed,
  });
}
