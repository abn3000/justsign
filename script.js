/* ============================================================================
   JustSign — fingerspelling rhythm game
   ----------------------------------------------------------------------------
   Sections:
     1. Sign definitions + procedural SVG hand diagrams
     2. Word lists
     3. Gemini API helpers (optional — user supplies their own AI Studio key)
     4. MediaPipe HandLandmarker setup + finger-state classifier
     5. Rhythm engine (note sequence, timing windows, scoring)
     6. Canvas rendering for the falling lane
     7. UI wiring / screen flow

   IMPORTANT NOTE ON ACCURACY: sign recognition here is a light-weight,
   hand-tuned heuristic classifier built directly on MediaPipe's 21 hand
   landmarks — NOT a trained ML model. It supports a deliberately-chosen
   subset of the ASL manual alphabet (the letters whose static handshapes are
   most distinguishable by simple geometry): A B C D F I L O U V W Y.
   Letters that require motion (J, Z) or very fine distinctions (M N S T E
   K P Q R X G H) are left out because a rules-based approach can't reliably
   tell them apart from a single frame. Real-time gameplay needs this to run
   at 30-60fps locally, which is why we don't call a cloud model per frame —
   Gemini is used instead for things that *aren't* on the timing-critical
   path (see section 3).
   ============================================================================ */

import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ---------------------------------------------------------------------------
   1. SIGN DEFINITIONS
   Each letter has a target feature vector used for nearest-neighbour
   classification, plus a simple pose descriptor used to draw the schematic
   hand diagram. curl: 0 = fully curled, 1 = fully extended, per finger
   [thumb, index, middle, ring, pinky]. angle (deg) is only used for the
   thumb/finger drawing on non-extended fingers to fan them out naturally.
--------------------------------------------------------------------------- */
const SIGNS = {
  A: { curl:[0.55,0,0,0,0],    thumbIndexDist:0.34, spreadIM:0.15, spreadMR:0.15, weight:{tid:1.4} },
  B: { curl:[0,1,1,1,1],       thumbIndexDist:0.30, spreadIM:0.14, spreadMR:0.14 },
  C: { curl:[0.4,0.4,0.4,0.4,0.4], thumbIndexDist:0.55, spreadIM:0.22, spreadMR:0.22, weight:{tid:1.6} },
  D: { curl:[0.25,1,0,0,0],    thumbIndexDist:0.40, spreadIM:0.30, spreadMR:0.15 },
  F: { curl:[0.3,0.3,1,1,1],   thumbIndexDist:0.14, spreadIM:0.30, spreadMR:0.14, weight:{tid:1.6} },
  I: { curl:[0,0,0,0,1],       thumbIndexDist:0.30, spreadIM:0.15, spreadMR:0.15 },
  L: { curl:[1,1,0,0,0],       thumbIndexDist:0.72, spreadIM:0.15, spreadMR:0.15, weight:{tid:1.4} },
  O: { curl:[0.35,0.35,0.35,0.35,0.35], thumbIndexDist:0.12, spreadIM:0.15, spreadMR:0.15, weight:{tid:1.8} },
  U: { curl:[0,1,1,0,0],       thumbIndexDist:0.30, spreadIM:0.12, spreadMR:0.15, weight:{sim:1.6} },
  V: { curl:[0,1,1,0,0],       thumbIndexDist:0.30, spreadIM:0.42, spreadMR:0.15, weight:{sim:1.6} },
  W: { curl:[0,1,1,1,0],       thumbIndexDist:0.30, spreadIM:0.34, spreadMR:0.30 },
  Y: { curl:[1,0,0,0,1],       thumbIndexDist:0.65, spreadIM:0.15, spreadMR:0.55 },
};
const SUPPORTED_LETTERS = Object.keys(SIGNS);

function drawHandSVG(letter){
  const def = SIGNS[letter];
  const svg = document.getElementById("sign-diagram");
  if(!def){ svg.innerHTML = ""; return; }
  const [t,i,m,r,p] = def.curl;
  // base positions: palm centered, fingers fan out upward from knuckle line
  const palm = `<rect x="70" y="120" width="60" height="70" rx="22" fill="none" stroke="var(--ink)" stroke-width="4"/>`;
  const fingers = [
    { name:'pinky',  x:78,  curl:p, len:46, ang:-18 },
    { name:'ring',   x:92,  curl:r, len:58, ang:-6 },
    { name:'middle', x:106, curl:m, len:64, ang:4 },
    { name:'index',  x:120, curl:i, len:56, ang:14 },
  ];
  let fingerPaths = "";
  fingers.forEach(f=>{
    const extendedLen = f.len;
    const curledLen = f.len*0.28;
    const L = curledLen + (extendedLen-curledLen)*f.curl;
    const rad = f.ang * Math.PI/180;
    const x1 = f.x, y1 = 122;
    const x2 = x1 + Math.sin(rad)*L*0.3;
    const y2 = y1 - L;
    const strokeColor = f.curl > 0.6 ? "var(--gold)" : "var(--ink-dim)";
    fingerPaths += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${strokeColor}" stroke-width="9" stroke-linecap="round"/>`;
  });
  // thumb: angle sweeps outward based on curl amount (0=tucked, 1=fully out to the side)
  const thumbSweep = 20 + t*70; // degrees from vertical palm edge
  const trad = thumbSweep * Math.PI/180;
  const tx1 = 70, ty1 = 150;
  const tlen = 30 + t*26;
  const tx2 = tx1 - Math.sin(trad)*tlen;
  const ty2 = ty1 - Math.cos(trad)*tlen*0.3 - 6;
  const thumbColor = t > 0.6 ? "var(--gold)" : "var(--ink-dim)";
  const thumbPath = `<line x1="${tx1}" y1="${ty1}" x2="${tx2}" y2="${ty2}" stroke="${thumbColor}" stroke-width="10" stroke-linecap="round"/>`;

  svg.innerHTML = `${palm}${fingerPaths}${thumbPath}`;
}

/* ---------------------------------------------------------------------------
   2. WORD LISTS (fallback, all letters drawn only from SUPPORTED_LETTERS)
--------------------------------------------------------------------------- */
const FALLBACK_WORDS = [
  "BOLD","WOLF","COLD","WILD","LOUD","OWL","FLY","VIA","LID","DIAL",
  "CLIFF","DOLL","FOWL","FOIL","VIVID","DAILY","DIVA","AWFUL","FLOOD",
  "FOLIO","BUY","COW","WOOL","IDOL","LOYAL","VOID","FLU","DUO"
];

/* ---------------------------------------------------------------------------
   3. GEMINI API HELPERS
   Gameplay scoring never depends on these — they're purely additive:
   generating themed word lists before a round, and a short generated
   commentary line after the round. Both are optional and skipped silently
   if no key is provided or the request fails.
--------------------------------------------------------------------------- */
const GEMINI_MODEL = "gemini-2.5-flash";

async function geminiGenerate(apiKey, prompt){
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.9, maxOutputTokens: 300 }
    })
  });
  if(!res.ok) throw new Error(`Gemini request failed: ${res.status}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.map(p=>p.text||"").join("") || "";
  return text.trim();
}

async function geminiThemedWords(apiKey, topic){
  const prompt = `Give me exactly 10 short English words (3-7 letters each) related to the theme "${topic}". ` +
    `Every letter in every word MUST come only from this set: ${SUPPORTED_LETTERS.join(", ")} (case-insensitive). ` +
    `No proper nouns. Reply with ONLY a JSON array of uppercase strings, nothing else, e.g. ["WORD","WORD2"].`;
  const raw = await geminiGenerate(apiKey, prompt);
  const cleaned = raw.replace(/```json|```/g, "").trim();
  const arr = JSON.parse(cleaned);
  const allowed = new Set(SUPPORTED_LETTERS);
  const valid = arr
    .map(w=>String(w).toUpperCase())
    .filter(w=>/^[A-Z]+$/.test(w) && [...w].every(ch=>allowed.has(ch)));
  if(valid.length === 0) throw new Error("Gemini returned no usable words");
  return valid;
}

async function geminiCommentary(apiKey, stats){
  const prompt = `You're a witty rhythm-game announcer. A player just finished a fingerspelling ` +
    `rhythm game called JustSign with: score ${stats.score}, accuracy ${stats.accuracy}%, ` +
    `best combo ${stats.bestCombo}, perfects ${stats.perfects}, goods ${stats.goods}, misses ${stats.misses}. ` +
    `Write 2 short, upbeat, specific sentences of commentary (no markdown, no emoji spam, max 1 emoji).`;
  return await geminiGenerate(apiKey, prompt);
}

/* ---------------------------------------------------------------------------
   4. MEDIAPIPE HAND TRACKING + CLASSIFIER
--------------------------------------------------------------------------- */
let handLandmarker = null;
let latestLandmarks = null; // 21 points of the most recently seen hand, or null

async function initHandLandmarker(){
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

// Extract a normalized feature vector from 21 raw landmarks.
function extractFeatures(lm){
  const wrist = lm[0];
  const scale = dist(wrist, lm[9]) || 1; // wrist -> middle MCP, used to normalize for hand size/distance from camera

  function fingerCurl(mcpI, pipI, tipI){
    const extendedRatio = dist(wrist, lm[tipI]) / (dist(wrist, lm[pipI]) || 1);
    // ~1.0 when curled (tip no farther than pip), grows toward ~1.6-1.8 when straight
    const curl = Math.max(0, Math.min(1, (extendedRatio - 0.95) / 0.55));
    return curl;
  }
  const thumbOpen = dist(lm[4], lm[17]) / (dist(lm[2], lm[17]) || 1);
  const thumbCurl = Math.max(0, Math.min(1, (thumbOpen - 0.8) / 0.9));

  const curl = [
    thumbCurl,
    fingerCurl(5,6,8),
    fingerCurl(9,10,12),
    fingerCurl(13,14,16),
    fingerCurl(17,18,20),
  ];

  const thumbIndexDist = dist(lm[4], lm[8]) / scale;
  const spreadIM = dist(lm[8], lm[12]) / scale;
  const spreadMR = dist(lm[12], lm[16]) / scale;

  return { curl, thumbIndexDist, spreadIM, spreadMR };
}

function classifySign(features){
  let best = null, bestDist = Infinity;
  for(const [letter, def] of Object.entries(SIGNS)){
    const wTid = def.weight?.tid ?? 1;
    const wSim = def.weight?.sim ?? 1;
    let d = 0;
    for(let k=0;k<5;k++){
      const diff = features.curl[k] - def.curl[k];
      d += diff*diff;
    }
    d += wTid * Math.pow(features.thumbIndexDist - def.thumbIndexDist, 2);
    d += wSim * Math.pow(features.spreadIM - def.spreadIM, 2);
    d += Math.pow(features.spreadMR - def.spreadMR, 2);
    if(d < bestDist){ bestDist = d; best = letter; }
  }
  // convert squared distance to a rough 0-1 confidence; tuned empirically
  const confidence = Math.max(0, 1 - bestDist/0.9);
  return { letter: best, confidence };
}

/* ---------------------------------------------------------------------------
   5. RHYTHM ENGINE
--------------------------------------------------------------------------- */
const DIFFICULTY = {
  easy:   { travelMs:3000, spawnGapMs:1500, perfectMs:220, goodMs:550, noteCount:16 },
  normal: { travelMs:2400, spawnGapMs:1150, perfectMs:150, goodMs:380, noteCount:20 },
  hard:   { travelMs:1850, spawnGapMs:900,  perfectMs:100, goodMs:260, noteCount:24 },
};

const state = {
  running:false,
  diff:null,
  notes:[],           // {letter, spawnTime, hitTime, judged, result}
  startTime:0,
  score:0,
  combo:0,
  bestCombo:0,
  perfects:0, goods:0, misses:0,
  wordList:null,
};

function buildNoteSequence(mode, diffKey, wordSource){
  const cfg = DIFFICULTY[diffKey];
  const letters = [];
  if(mode === "words"){
    const words = wordSource && wordSource.length ? wordSource : FALLBACK_WORDS;
    const shuffled = [...words].sort(()=>Math.random()-0.5);
    for(const w of shuffled){
      if(letters.length >= cfg.noteCount) break;
      for(const ch of w){
        if(letters.length >= cfg.noteCount) break;
        letters.push(ch);
      }
      letters.push(null); // small gap marker between words
    }
  } else {
    let last = null;
    for(let i=0;i<cfg.noteCount;i++){
      let choice;
      do { choice = SUPPORTED_LETTERS[Math.floor(Math.random()*SUPPORTED_LETTERS.length)]; }
      while(choice === last);
      last = choice;
      letters.push(choice);
    }
  }

  const notes = [];
  let t = cfg.travelMs + 400; // lead-in so the very first note is visible falling from the top
  for(const ch of letters){
    if(ch === null){ t += cfg.spawnGapMs*0.6; continue; }
    notes.push({ letter:ch, hitTime:t, judged:false, result:null });
    t += cfg.spawnGapMs;
  }
  return notes;
}

function currentActiveNote(now){
  for(const n of state.notes){
    if(!n.judged) return n;
  }
  return null;
}

function judge(note, result){
  note.judged = true;
  note.result = result;
  if(result === "perfect"){
    state.score += 100 + state.combo*2;
    state.combo++; state.perfects++;
  } else if(result === "good"){
    state.score += 60 + state.combo;
    state.combo++; state.goods++;
  } else {
    state.combo = 0; state.misses++;
  }
  state.bestCombo = Math.max(state.bestCombo, state.combo);
  flashJudgement(result);
}

function updateEngine(now){
  const cfg = state.diff;
  const active = currentActiveNote(now);
  if(!active){
    if(state.notes.length && state.notes.every(n=>n.judged)){
      endGame();
    }
    return;
  }
  const timeDiff = now - active.hitTime;

  if(timeDiff > cfg.goodMs){
    judge(active, "miss");
    return;
  }

  if(timeDiff >= -cfg.goodMs && latestLandmarks){
    const features = extractFeatures(latestLandmarks);
    const guess = classifySign(features);
    updateDetectRing(guess.letter === active.letter && guess.confidence > 0.35 ? "hover" : "idle");
    if(guess.letter === active.letter && guess.confidence > 0.35){
      const abs = Math.abs(timeDiff);
      judge(active, abs <= cfg.perfectMs ? "perfect" : "good");
    }
  } else {
    updateDetectRing("idle");
  }

  // refresh the "make this sign" panel to reflect the active note
  if(target_letter_big.textContent !== active.letter){
    target_letter_big.textContent = active.letter;
    drawHandSVG(active.letter);
  }
}

function flashJudgement(result){
  judgementValue.textContent = result === "perfect" ? "PERFECT" : result === "good" ? "GOOD" : "MISS";
  judgementValue.className = `hud-value judgement ${result}`;
  scoreValue.textContent = state.score;
  comboValue.textContent = state.combo;
  const totalJudged = state.perfects + state.goods + state.misses;
  const acc = totalJudged ? ((state.perfects + state.goods*0.6) / totalJudged * 100) : 100;
  accuracyValue.textContent = `${acc.toFixed(0)}%`;
  detectRing.classList.remove("correct","wrong");
  detectRing.classList.add(result === "miss" ? "wrong" : "correct");
  setTimeout(()=>detectRing.classList.remove("correct","wrong"), 220);
}

function updateDetectRing(mode){
  detectRing.classList.toggle("near", mode === "hover");
}

/* ---------------------------------------------------------------------------
   6. LANE CANVAS RENDERING
--------------------------------------------------------------------------- */
const laneCanvas = document.getElementById("lane-canvas");
const laneCtx = laneCanvas.getContext("2d");

function resizeLaneCanvas(){
  const panel = laneCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  laneCanvas.width = panel.clientWidth * dpr;
  laneCanvas.height = panel.clientHeight * dpr;
  laneCanvas.style.width = panel.clientWidth + "px";
  laneCanvas.style.height = panel.clientHeight + "px";
  laneCtx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener("resize", resizeLaneCanvas);

function renderLane(now){
  const w = laneCanvas.clientWidth, h = laneCanvas.clientHeight;
  laneCtx.clearRect(0,0,w,h);

  const hitY = h - 90;
  const laneX = w/2;

  // signature element: spotlight beam over the lane, brightest near the hit line
  const beam = laneCtx.createLinearGradient(0,0,0,hitY+40);
  beam.addColorStop(0, "rgba(242,193,78,0.02)");
  beam.addColorStop(0.75, "rgba(242,193,78,0.05)");
  beam.addColorStop(1, "rgba(242,193,78,0.16)");
  laneCtx.fillStyle = beam;
  laneCtx.beginPath();
  laneCtx.moveTo(laneX-70, 0);
  laneCtx.lineTo(laneX+70, 0);
  laneCtx.lineTo(laneX+130, hitY+40);
  laneCtx.lineTo(laneX-130, hitY+40);
  laneCtx.closePath();
  laneCtx.fill();

  // hit ring, pulses when a note is in its perfect window
  const active = state.running ? currentActiveNote(now) : null;
  let pulse = 0;
  if(active){
    const diff = Math.abs(now - active.hitTime);
    if(diff < state.diff.perfectMs) pulse = 1 - diff/state.diff.perfectMs;
  }
  const ringR = 46 + pulse*10;
  laneCtx.beginPath();
  laneCtx.arc(laneX, hitY, ringR, 0, Math.PI*2);
  laneCtx.strokeStyle = pulse > 0 ? "#3FBFB0" : "rgba(237,235,245,0.35)";
  laneCtx.lineWidth = 4;
  laneCtx.stroke();
  laneCtx.beginPath();
  laneCtx.arc(laneX, hitY, ringR-14, 0, Math.PI*2);
  laneCtx.strokeStyle = "rgba(237,235,245,0.18)";
  laneCtx.lineWidth = 2;
  laneCtx.stroke();

  if(!state.running) return;

  // falling notes
  const cfg = state.diff;
  for(const note of state.notes){
    if(note.judged) continue;
    const spawnTime = note.hitTime - cfg.travelMs;
    const progress = (now - spawnTime) / cfg.travelMs;
    if(progress < -0.02 || progress > 1.15) continue;
    const y = progress * hitY;
    if(y < -40) continue;

    laneCtx.beginPath();
    laneCtx.arc(laneX, y, 26, 0, Math.PI*2);
    laneCtx.fillStyle = "#251F3D";
    laneCtx.strokeStyle = "#F2C14E";
    laneCtx.lineWidth = 3;
    laneCtx.fill();
    laneCtx.stroke();

    laneCtx.fillStyle = "#EDEBF5";
    laneCtx.font = "700 26px 'Baloo 2', sans-serif";
    laneCtx.textAlign = "center";
    laneCtx.textBaseline = "middle";
    laneCtx.fillText(note.letter, laneX, y+2);
  }
}

/* ---------------------------------------------------------------------------
   7. UI WIRING / SCREEN FLOW
--------------------------------------------------------------------------- */
const startScreen = document.getElementById("start-screen");
const gameScreen = document.getElementById("game-screen");
const resultsScreen = document.getElementById("results-screen");

const startBtn = document.getElementById("start-btn");
const cameraStatus = document.getElementById("camera-status");
const modeSelect = document.getElementById("mode-select");
const difficultySelect = document.getElementById("difficulty-select");
const geminiKeyInput = document.getElementById("gemini-key");
const geminiTopicBtn = document.getElementById("gemini-topic-btn");

const scoreValue = document.getElementById("score-value");
const comboValue = document.getElementById("combo-value");
const judgementValue = document.getElementById("judgement-value");
const accuracyValue = document.getElementById("accuracy-value");
const target_letter_big = document.getElementById("target-letter-big");
const signCaption = document.getElementById("sign-caption");
const detectRing = document.getElementById("detect-ring");
const quitBtn = document.getElementById("quit-btn");
const replayBtn = document.getElementById("replay-btn");

const webcamVideo = document.getElementById("webcam");
const landmarkCanvas = document.getElementById("landmark-canvas");
const landmarkCtx = landmarkCanvas.getContext("2d");

let cameraReady = false, modelReady = false, gameLoopHandle = null;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

function drawLandmarks(lm){
  const w = landmarkCanvas.clientWidth, h = landmarkCanvas.clientHeight;
  landmarkCanvas.width = w; landmarkCanvas.height = h;
  landmarkCtx.clearRect(0,0,w,h);
  if(!lm) return;
  landmarkCtx.strokeStyle = "rgba(63,191,176,0.85)";
  landmarkCtx.lineWidth = 2;
  for(const [a,b] of HAND_CONNECTIONS){
    landmarkCtx.beginPath();
    landmarkCtx.moveTo(lm[a].x*w, lm[a].y*h);
    landmarkCtx.lineTo(lm[b].x*w, lm[b].y*h);
    landmarkCtx.stroke();
  }
  landmarkCtx.fillStyle = "#F2C14E";
  for(const p of lm){
    landmarkCtx.beginPath();
    landmarkCtx.arc(p.x*w, p.y*h, 3.5, 0, Math.PI*2);
    landmarkCtx.fill();
  }
}

async function setup(){
  drawHandSVG("A");
  try{
    await initHandLandmarker();
    modelReady = true;
  }catch(e){
    console.error(e);
    cameraStatus.textContent = "Couldn't load the hand-tracking model. Check your connection and reload.";
    return;
  }
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video:{ width:640, height:480 }, audio:false });
    webcamVideo.srcObject = stream;
    await new Promise(res=>{ webcamVideo.onloadedmetadata = ()=>{ webcamVideo.play(); res(); }; });
    cameraReady = true;
  }catch(e){
    console.error(e);
    cameraStatus.textContent = "Camera access was blocked. Allow camera permissions and reload to play.";
    return;
  }
  cameraStatus.textContent = "Camera and hand tracking ready.";
  startBtn.disabled = false;
  startBtn.textContent = "Start song";
  detectionLoop();
}

function detectionLoop(){
  function loop(){
    if(handLandmarker && webcamVideo.readyState >= 2){
      const result = handLandmarker.detectForVideo(webcamVideo, performance.now());
      if(result.landmarks && result.landmarks.length){
        latestLandmarks = result.landmarks[0];
      } else {
        latestLandmarks = null;
      }
      // draw on the (un-mirrored) coordinate space; CSS mirrors both video+canvas together
      drawLandmarks(latestLandmarks);
    }
    requestAnimationFrame(loop);
  }
  loop();
}

geminiKeyInput.addEventListener("input", ()=>{
  geminiTopicBtn.disabled = geminiKeyInput.value.trim().length === 0;
});

geminiTopicBtn.addEventListener("click", async ()=>{
  const key = geminiKeyInput.value.trim();
  if(!key) return;
  const topic = prompt("What theme should Gemini pick words from? (e.g. \"food\", \"weather\", \"animals\")", "animals");
  if(!topic) return;
  geminiTopicBtn.disabled = true;
  geminiTopicBtn.textContent = "Asking Gemini…";
  try{
    state.wordList = await geminiThemedWords(key, topic);
    modeSelect.value = "words";
    geminiTopicBtn.textContent = `✓ "${topic}" words ready`;
  }catch(e){
    console.error(e);
    geminiTopicBtn.textContent = "Couldn't fetch — using built-ins";
    state.wordList = null;
    setTimeout(()=>{ geminiTopicBtn.textContent = "🎲 Themed words"; }, 2200);
  }finally{
    geminiTopicBtn.disabled = false;
  }
});

startBtn.addEventListener("click", ()=>{
  if(!cameraReady || !modelReady) return;
  startGame();
});
replayBtn.addEventListener("click", ()=>{
  resultsScreen.classList.add("hidden");
  startScreen.classList.remove("hidden");
});
quitBtn.addEventListener("click", ()=> endGame());

function startGame(){
  const diffKey = difficultySelect.value;
  const mode = modeSelect.value;
  state.diff = DIFFICULTY[diffKey];
  state.notes = buildNoteSequence(mode, diffKey, state.wordList);
  state.startTime = performance.now();
  state.score = 0; state.combo = 0; state.bestCombo = 0;
  state.perfects = 0; state.goods = 0; state.misses = 0;
  state.running = true;

  scoreValue.textContent = "0";
  comboValue.textContent = "0";
  accuracyValue.textContent = "100%";
  judgementValue.textContent = "—";
  judgementValue.className = "hud-value judgement";

  startScreen.classList.add("hidden");
  resultsScreen.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  resizeLaneCanvas();

  if(state.notes.length){
    target_letter_big.textContent = state.notes[0].letter;
    drawHandSVG(state.notes[0].letter);
    signCaption.textContent = `Fingerspell the letter shown — note ${1} of ${state.notes.length}`;
  }

  cancelAnimationFrame(gameLoopHandle);
  function tick(){
    const now = performance.now() - state.startTime;
    if(state.running){
      updateEngine(now);
      const remaining = state.notes.filter(n=>!n.judged).length;
      signCaption.textContent = `${remaining} letter${remaining===1?"":"s"} to go`;
    }
    renderLane(now);
    if(state.running){
      gameLoopHandle = requestAnimationFrame(tick);
    }
  }
  tick();
}

async function endGame(){
  if(!state.running) return;
  state.running = false;
  cancelAnimationFrame(gameLoopHandle);

  const totalJudged = state.perfects + state.goods + state.misses;
  const accuracy = totalJudged ? ((state.perfects + state.goods*0.6) / totalJudged * 100) : 0;

  document.getElementById("final-score").textContent = state.score;
  document.getElementById("final-combo").textContent = state.bestCombo;
  document.getElementById("final-accuracy").textContent = `${accuracy.toFixed(0)}%`;
  document.getElementById("final-perfects").textContent = state.perfects;

  const rank = accuracy >= 95 ? "Encore!" : accuracy >= 80 ? "Great Set" : accuracy >= 60 ? "Solid Show" : "Keep Rehearsing";
  document.getElementById("final-rank").textContent = rank;

  gameScreen.classList.add("hidden");
  resultsScreen.classList.remove("hidden");

  const commentaryEl = document.getElementById("gemini-commentary");
  const key = geminiKeyInput.value.trim();
  if(key){
    commentaryEl.classList.remove("hidden");
    commentaryEl.textContent = "Gemini is composing a note on your performance…";
    try{
      const text = await geminiCommentary(key, {
        score: state.score, accuracy: accuracy.toFixed(0),
        bestCombo: state.bestCombo, perfects: state.perfects,
        goods: state.goods, misses: state.misses
      });
      commentaryEl.textContent = text;
    }catch(e){
      console.error(e);
      commentaryEl.classList.add("hidden");
    }
  } else {
    commentaryEl.classList.add("hidden");
  }
}

setup();
