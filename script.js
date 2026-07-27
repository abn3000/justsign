const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const WORDS = [
  // 3 letters
  "CAT","DOG","SUN","RUN","BIG","RED","SKY","JOY","TOP","FUN","ICE","BEE","OWL","FOX","ARM",
  // 4 letters
  "CODE","GAME","LOVE","STAR","MOON","WIND","LEAF","BIRD","FISH","GOLD","WAVE","ROCK","JUMP","BOLD","FIRE",
  // 5 letters
  "HELLO","WORLD","HAPPY","MUSIC","DANCE","QUICK","BRAVE","SMILE","LIGHT","PIZZA","OCEAN","TIGER","SPARK","CLOUD","MAGIC","STORM","FLAME","RIVER","GHOST","PLANT","SUGAR","MANGO","EARTH","PEACE","DREAM",
  // 6 letters
  "PLANET","ROCKET","SILVER","BRIGHT","FRIEND","GARDEN","WINTER","SUMMER","PURPLE","ORANGE","SPIRIT","CASTLE","BREEZE","JUNGLE","VOYAGE",
  // 7 letters
  "RAINBOW","JOURNEY","FREEDOM","HORIZON","CRYSTAL","PICTURE","AMAZING","THUNDER","HARMONY","CAPTAIN",
  // 8 letters
  "SUNSHINE","STARLIGHT","MOUNTAIN","BUTTERFLY","ADVENTURE","FANTASTIC","DIAMONDS","CREATIVE"
].filter(w=>w.length>=3 && w.length<=8);

let dataset = {}; LETTERS.forEach(l=>dataset[l]=[]);

/* KNN Math Model */
const FINGER_JOINTS = { thumb:[1,2,3,4], index:[5,6,7,8], middle:[9,10,11,12], ring:[13,14,15,16], pinky:[17,18,19,20] };
const FINGERTIPS = [4,8,12,16,20];
function vec3(a,b){return {x:b.x-a.x,y:b.y-a.y,z:b.z-a.z};}
function len3(v){return Math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)||1e-9;}
function dot3(a,b){return a.x*b.x+a.y*b.y+a.z*b.z;}
function angleBetween(a,b){const c=dot3(a,b)/(len3(a)*len3(b));return Math.acos(Math.max(-1,Math.min(1,c)));}

function extractFeatures(lm){
  const wrist = lm[0];
  const scale = len3(vec3(wrist, lm[9])) || 1e-6;
  const feats = [];
  Object.values(FINGER_JOINTS).forEach(chain=>{
    for(let i=1;i<chain.length-1;i++){
      const p0=lm[chain[i-1]],p1=lm[chain[i]],p2=lm[chain[i+1]];
      feats.push(angleBetween(vec3(p1,p0),vec3(p1,p2))/Math.PI);
    }
  });
  FINGERTIPS.forEach(tip=>feats.push(len3(vec3(wrist,lm[tip]))/scale));
  for(let i=0;i<FINGERTIPS.length;i++)for(let j=i+1;j<FINGERTIPS.length;j++)
    feats.push(len3(vec3(lm[FINGERTIPS[i]],lm[FINGERTIPS[j]]))/scale);
  const palmZ=(lm[0].z+lm[5].z+lm[9].z+lm[13].z+lm[17].z)/5;
  FINGERTIPS.forEach(tip=>feats.push((lm[tip].z-palmZ)/scale));
  for(let i=0;i<FINGERTIPS.length;i++)for(let j=i+1;j<FINGERTIPS.length;j++)
    feats.push((lm[FINGERTIPS[i]].z-lm[FINGERTIPS[j]].z)/scale);
  const thumbTip=lm[4];
  [5,9,13,17].forEach(mcp=>feats.push(len3(vec3(thumbTip,lm[mcp]))/scale));
  const palmCenter={x:(lm[0].x+lm[5].x+lm[9].x+lm[13].x+lm[17].x)/5,y:(lm[0].y+lm[5].y+lm[9].y+lm[13].y+lm[17].y)/5,z:(lm[0].z+lm[5].z+lm[9].z+lm[13].z+lm[17].z)/5};
  let openness=0; FINGERTIPS.forEach(tip=>openness+=len3(vec3(palmCenter,lm[tip]))/scale);
  feats.push(openness/5);
  return feats;
}
function euclidean(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return Math.sqrt(s);}
function classify(feat,k=7){
  const all=[];
  for(const l of LETTERS)for(const f of dataset[l])all.push({l,d:euclidean(feat,f)});
  if(all.length===0)return null;
  all.sort((a,b)=>a.d-b.d);
  const top=all.slice(0,Math.min(k,all.length));
  const votes={};
  top.forEach(t=>{const w=1/(t.d+1e-3);votes[t.l]=(votes[t.l]||0)+w;});
  let best=null,bestW=-1,totalW=0;
  Object.entries(votes).forEach(([l,w])=>{totalW+=w;if(w>bestW){bestW=w;best=l;}});
  return {letter:best,confidence:bestW/totalW};
}

const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function drawHand(ctx,canvas,video,lm){
  canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!lm)return;
  ctx.strokeStyle='#00f0ff'; ctx.lineWidth=3;
  CONNECTIONS.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(lm[a].x*canvas.width,lm[a].y*canvas.height);ctx.lineTo(lm[b].x*canvas.width,lm[b].y*canvas.height);ctx.stroke();});
  ctx.fillStyle='#ff2e9a';
  lm.forEach(p=>{ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);ctx.fill();});
}

/* Baked-in team dataset — one file per teammate, merged together at load time.
   Add or rename entries here to match whatever files land in /training-data/. */
const DEFAULT_DATASET_FILES = [
  'training-data/teammate1.json',
  'training-data/teammate2.json',
  'training-data/teammate3.json',
  'training-data/teammate4.json'
];

let bakedDataset = {}; LETTERS.forEach(l=>bakedDataset[l]=[]);   // from /training-data/*.json
let personalDataset = {}; LETTERS.forEach(l=>personalDataset[l]=[]); // this browser only

function rebuildDataset(){ LETTERS.forEach(l=>{ dataset[l] = bakedDataset[l].concat(personalDataset[l]); }); }

async function loadDefaultDatasets(){
  let filesLoaded=0;
  for(const path of DEFAULT_DATASET_FILES){
    try{
      const res = await fetch(path);
      if(!res.ok) continue;
      const parsed = await res.json();
      LETTERS.forEach(l=>{ if(parsed[l] && parsed[l].length) bakedDataset[l]=bakedDataset[l].concat(parsed[l]); });
      filesLoaded++;
    }catch(e){ /* file missing or not JSON yet — skip it */ }
  }
  return filesLoaded;
}

/* Personal data (this browser only) — never mixed with baked-in data when saving,
   so reloading never double-counts the team's samples. */
async function saveData(){ try{ localStorage.setItem('asl_dataset_v1', JSON.stringify(personalDataset)); }catch(e){} }
async function loadData(){
  const filesLoaded = await loadDefaultDatasets();
  try{
    const raw = localStorage.getItem('asl_dataset_v1');
    if(raw){
      const parsed = JSON.parse(raw);
      LETTERS.forEach(l=>{ if(parsed[l]) personalDataset[l]=parsed[l]; });
    }
  }catch(e){}
  rebuildDataset();
  const total = LETTERS.reduce((s,l)=>s+dataset[l].length,0);
  const personalTotal = LETTERS.reduce((s,l)=>s+personalDataset[l].length,0);
  document.getElementById('loadMsg').textContent = total
    ? `Loaded ${total} samples (${filesLoaded} team file${filesLoaded===1?'':'s'}${personalTotal?` + ${personalTotal} of your own`:''}).`
    : 'No training data yet — capture some samples or add team files to /training-data/.';
  renderLetterGrid(); renderLetterGraph();
}

/* Manual Training */
const letterSelect=document.getElementById('letterSelect');
const letterGrid=document.getElementById('letterGrid');
let selectedLetter='A';
LETTERS.forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=l;letterSelect.appendChild(o);});
letterSelect.onchange=()=>{selectedLetter=letterSelect.value;renderLetterGrid();};

function renderLetterGrid(){
  letterGrid.innerHTML='';
  LETTERS.forEach(l=>{
    const n=dataset[l].length;
    const d=document.createElement('div');
    d.className='lb'+(l===selectedLetter?' selected':'')+(n>0?' has-data':'');
    d.innerHTML=`${l}<span class="cnt">${n}</span>`;
    d.onclick=()=>{selectedLetter=l;letterSelect.value=l;renderLetterGrid();};
    letterGrid.appendChild(d);
  });
}
function renderLetterGraph(){
  const wrap=document.getElementById('letterGraph');
  wrap.innerHTML='';
  const max=Math.max(1,...LETTERS.map(l=>dataset[l].length));
  LETTERS.forEach(l=>{
    const n=dataset[l].length;
    const row=document.createElement('div'); row.className='lg-row';
    row.innerHTML=`<span class="lg-l">${l}</span><div class="lg-bar"><div style="width:${n?Math.max(6,(n/max)*100):0}%"></div></div>`;
    wrap.appendChild(row);
  });
}
renderLetterGrid();

let tvideo=document.getElementById('tvideo'), toverlay=document.getElementById('toverlay'), tctx=toverlay.getContext('2d');
let trainLatestLM=null, trainCamera=null, trainStarted=false;
const trainHands = new Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`});
trainHands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:0.7,minTrackingConfidence:0.7});
trainHands.onResults(res=>{
  if(res.multiHandLandmarks && res.multiHandLandmarks.length>0){
    trainLatestLM=res.multiHandLandmarks[0];
    drawHand(tctx,toverlay,tvideo,trainLatestLM);
    document.getElementById('captureBtn').disabled=false;
    document.getElementById('burstBtn').disabled=false;
  } else {
    trainLatestLM=null;
    drawHand(tctx,toverlay,tvideo,null);
    document.getElementById('captureBtn').disabled=true;
    document.getElementById('burstBtn').disabled=true;
  }
});
document.getElementById('trainCamBtn').onclick=async()=>{
  if(trainStarted)return;
  const btn=document.getElementById('trainCamBtn'); btn.textContent='Starting…'; btn.disabled=true;
  try{
    trainCamera=new Camera(tvideo,{onFrame:async()=>{await trainHands.send({image:tvideo});},width:640,height:480});
    await trainCamera.start();
    trainStarted=true; btn.textContent='Camera running';
    document.getElementById('captureStatus').textContent='Hold a sign steady, then capture.';
  }catch(e){ btn.textContent='Retry camera'; btn.disabled=false; document.getElementById('captureStatus').textContent='Camera error: '+e.message; }
};
function addSample(){
  if(!trainLatestLM) return false;
  const feat = extractFeatures(trainLatestLM);
  personalDataset[selectedLetter].push(feat);
  dataset[selectedLetter].push(feat);
  return true;
}
document.getElementById('captureBtn').onclick=()=>{
  if(addSample()){ document.getElementById('captureStatus').textContent=`Captured #${dataset[selectedLetter].length} for "${selectedLetter}".`; renderLetterGrid(); renderLetterGraph(); saveData(); }
  else document.getElementById('captureStatus').textContent='No hand detected.';
};
document.getElementById('burstBtn').onclick=()=>{
  let count=0;
  const iv=setInterval(()=>{
    if(addSample())count++;
    if(count>=20){ clearInterval(iv); document.getElementById('captureStatus').textContent=`Burst done: +${count} for "${selectedLetter}".`; renderLetterGrid(); renderLetterGraph(); saveData(); }
  },120);
  document.getElementById('captureStatus').textContent='Burst capturing…';
};
document.getElementById('clearLetterBtn').onclick=()=>{ personalDataset[selectedLetter]=[]; rebuildDataset(); renderLetterGrid(); renderLetterGraph(); saveData(); document.getElementById('captureStatus').textContent=`Cleared your samples for "${selectedLetter}" (team defaults kept).`; };
document.getElementById('clearAllBtn').onclick=()=>{ if(!confirm('Clear all of YOUR captured samples? (Team defaults will remain.)'))return; LETTERS.forEach(l=>personalDataset[l]=[]); rebuildDataset(); renderLetterGrid(); renderLetterGraph(); saveData(); document.getElementById('captureStatus').textContent='Cleared your samples (team defaults kept).'; };
document.getElementById('exportBtn').onclick=()=>{
  const blob = new Blob([JSON.stringify(personalDataset)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'training-data.json'; a.click();
  URL.revokeObjectURL(url);
  document.getElementById('captureStatus').textContent = 'Exported — rename the file (e.g. teammate2.json) and add it to /training-data/ in the repo.';
};

const trainModal=document.getElementById('trainModal');
document.getElementById('openTrainBtn').onclick=()=>trainModal.classList.add('show');
document.getElementById('closeTrainBtn').onclick=()=>trainModal.classList.remove('show');
trainModal.addEventListener('click',e=>{ if(e.target===trainModal) trainModal.classList.remove('show'); });

/* Signing Game */
let gvideo=document.getElementById('gvideo'), goverlay=document.getElementById('goverlay'), gctx=goverlay.getContext('2d');
let gameLatestLM=null, gameCamera=null, gameCamStarted=false;
const gameHands = new Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`});
gameHands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:0.7,minTrackingConfidence:0.7});
gameHands.onResults(res=>{
  if(res.multiHandLandmarks && res.multiHandLandmarks.length>0){
    gameLatestLM=res.multiHandLandmarks[0];
    drawHand(gctx,goverlay,gvideo,gameLatestLM);
  } else { gameLatestLM=null; drawHand(gctx,goverlay,gvideo,null); }
});

document.getElementById('startCamBtn').onclick=async()=>{
  if(gameCamStarted)return;
  const btn=document.getElementById('startCamBtn'); btn.textContent='Starting…'; btn.disabled=true;
  try{
    gameCamera=new Camera(gvideo,{onFrame:async()=>{await gameHands.send({image:gvideo});},width:640,height:480});
    await gameCamera.start();
    gameCamStarted=true; btn.textContent='Camera live';
    document.getElementById('playBtn').disabled=false;
    document.getElementById('playBtn').textContent='Start game';
  }catch(e){ btn.textContent='Retry camera'; btn.disabled=false; }
};

// difficulty
let letterDelay=4000;
document.querySelectorAll('.difficulty button').forEach(b=>{
  b.onclick=()=>{ document.querySelectorAll('.difficulty button').forEach(x=>x.classList.remove('sel')); b.classList.add('sel'); letterDelay=parseInt(b.dataset.d); };
});

let score=0, combo=0, bestCombo=0, hits=0, misses=0, wordsCleared=0;
let gameRunning=false;
let currentWord='', letterQueue=[], activeIndex=-1, holdAccum=0, requiredHold=420, missTimer=null;
const lane=document.getElementById('lane');
const hitzone=document.getElementById('hitzone');
const hzLetter=document.getElementById('hzLetter');

function updateHUD(){
  const scoreEl=document.getElementById('scoreVal');
  scoreEl.textContent=score;
  scoreEl.classList.remove('bump'); void scoreEl.offsetWidth; scoreEl.classList.add('bump');
  document.getElementById('sideScore').textContent=score;
  document.getElementById('comboVal').textContent=combo+'x combo';
  document.getElementById('sideBestCombo').textContent=bestCombo;
  document.getElementById('sideWords').textContent=wordsCleared;
  const total=hits+misses;
  document.getElementById('sideAcc').textContent = total? Math.round(hits/total*100)+'%' : '–';
  document.getElementById('sbFill').style.width = Math.min(100,(combo/15)*100)+'%';
}

function pickWord(){ return WORDS[Math.floor(Math.random()*WORDS.length)]; }

function renderWordLine(){
  const wl=document.getElementById('wordLine');
  wl.innerHTML='';
  currentWord.split('').forEach((ch,i)=>{
    const d=document.createElement('div');
    d.className='wletter'+(i<activeIndex?' done':i===activeIndex?' now':'');
    d.textContent=ch;
    wl.appendChild(d);
  });
  document.getElementById('wordVal').textContent=currentWord.split('').join(' · ');
}

function clearLane(){ lane.querySelectorAll('.tile').forEach(t=>t.remove()); }

function spawnWord(){
  currentWord = pickWord();
  activeIndex = -1;
  renderWordLine();
  clearLane();
  letterQueue = currentWord.split('').map((ch,i)=>({
    ch, spawnTime: performance.now() + (i+1)*letterDelay + 900,
    el:null, judged:false
  }));
  letterQueue.forEach((item,i)=>{
    const tile=document.createElement('div');
    tile.className='tile upcoming';
    tile.style.left='1100px';
    tile.innerHTML = `${item.ch}<div class="prog"><div></div></div>`;
    lane.appendChild(tile);
    item.el=tile;
  });
}

function floatScore(text,color,xpx){
  const f=document.createElement('div');
  f.className='floatpts'; f.textContent=text; f.style.color=color;
  f.style.left=(xpx||60)+'px'; f.style.top='40px';
  lane.appendChild(f);
  setTimeout(()=>f.remove(),700);
}

let lastFrameTime=performance.now();
function gameLoop(now){
  if(!gameRunning) return;
  const laneWidth = lane.getBoundingClientRect().width || 1000;
  const hzCenter = 22+42; // hitzone left+half width approx
  letterQueue.forEach((item,i)=>{
    if(item.judged) return;
    const t = now - (item.spawnTime - letterDelay - 900); // time since this tile started sliding
    const totalTravel = letterDelay + 900; // ms to go from spawn(right) to hitzone
    const timeToHit = item.spawnTime - now; // ms remaining until it's "due"
    // position: interpolate from spawnX (off-screen right) to hzCenter (at due time)
    const spawnX = laneWidth + 80; // safely past the visible edge, fully hidden until it slides in
    const progress = 1 - (timeToHit / totalTravel); // goes from 0 to 1 from spawn to "due"
    const clamped = Math.max(-0.15, Math.min(1.25, progress));
    const x = spawnX - clamped*(spawnX-hzCenter);
    item.el.style.left = x+'px';

    const isArmed = timeToHit < 900 && timeToHit > -900;
    if(isArmed){
      item.el.classList.remove('upcoming');
      if(activeIndex!==i){ activeIndex=i; renderWordLine(); }
      hitzone.classList.add('armed');
      hzLetter.textContent=item.ch;

      // check prediction hold
      let matched=false;
      if(gameLatestLM){
        const feat = extractFeatures(gameLatestLM);
        const result = classify(feat);
        if(result && result.letter===item.ch && result.confidence>0.42){ matched=true; }
      }
      if(matched){ holdAccum += (now-lastFrameTime); } else { holdAccum = Math.max(0,holdAccum-(now-lastFrameTime)*1.4); }
      const progBar = item.el.querySelector('.prog>div');
      if(progBar) progBar.style.width = Math.min(100,(holdAccum/requiredHold)*100)+'%';

      if(holdAccum>=requiredHold){
        item.judged=true;
        item.el.classList.add('hit');
        hits++; combo++; bestCombo=Math.max(bestCombo,combo);
        score += 100 + combo*10;
        floatScore('+'+(100+combo*10),'#39ff88',80);
        holdAccum=0;
        hitzone.classList.remove('armed');
        setTimeout(()=>{ if(item.el)item.el.remove(); },350);
        updateHUD();
      }
    }
    if(timeToHit < -900 && !item.judged){
      item.judged=true;
      item.el.classList.add('miss');
      misses++; combo=0;
      floatScore('MISS','#ff3860',80);
      hitzone.classList.remove('armed');
      holdAccum=0;
      setTimeout(()=>{ if(item.el)item.el.remove(); },350);
      updateHUD();
    }
  });

  if(letterQueue.length>0 && letterQueue.every(it=>it.judged)){
    wordsCleared++;
    updateHUD();
    letterQueue = []; // prevent double-trigger
    setTimeout(()=>{ if(gameRunning) spawnWord(); }, 700);
  }

  lastFrameTime = now;
  requestAnimationFrame(gameLoop);
}

document.getElementById('playBtn').onclick=()=>{
  document.getElementById('centerMsg').style.display='none';
  gameRunning=true;
  score=0;combo=0;bestCombo=0;hits=0;misses=0;wordsCleared=0;
  updateHUD();
  lastFrameTime=performance.now();
  spawnWord();
  requestAnimationFrame(gameLoop);
};

loadData();
