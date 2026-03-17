"use strict";
const METRICA_COUNTER_ID = 107726491;
function track(goal, params) {
  if (typeof ym === "undefined") return;
  ym(METRICA_COUNTER_ID, "reachGoal", goal, params ?? {});
}
function setVisitParams(params) {
  if (typeof ym === "undefined") return;
  ym(METRICA_COUNTER_ID, "params", params);
}
const _sessionStart = Date.now();
function trackSessionStart(abGroup) {
  const isReturning = Boolean(localStorage.getItem("tutorialSeen"));
  track("session_start", { is_returning: isReturning, ab_group: abGroup });
  setVisitParams({ ab_group: abGroup, is_returning: isReturning ? 1 : 0 });
  window.addEventListener("beforeunload", () => {
    const duration = Math.round((Date.now() - _sessionStart) / 1000);
    track("session_end", { duration_sec: duration });
  });
}
function trackLevelStart(level, themeLetter) {
  track("level_started", { level, theme_letter: themeLetter });
  setVisitParams({ level, level_status: "in_progress" });
}
function trackLevelComplete(level, hintsUsed, wordsTotal) {
  track("level_completed", {
    level,
    hints_used: hintsUsed,
    words_total: wordsTotal,
  });
  setVisitParams({ level_status: "completed", hints_used: hintsUsed });
}
function trackHintReceived(level, wordsFound, wordsTotal) {
  const completion_pct =
    wordsTotal > 0 ? Math.round((wordsFound / wordsTotal) * 100) : 0;
  track("hint_received", {
    level,
    words_found: wordsFound,
    words_total: wordsTotal,
    completion_pct,
  });
}
function trackDropOff(level, wordsFound, wordsTotal, hintsUsed) {
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
