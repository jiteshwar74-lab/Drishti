'use strict';

// ── OBJECT CLASSES ────────────────────────────────
const LABELS = {
    0: { name: 'person' },
    1: { name: 'firearm' },
    2: { name: 'knife' },
    3: { name: 'grenade' }
};
const PERSON_CLASS = 'person';
const WEAPON_CLASSES = new Set(['firearm', 'knife', 'grenade']);
// We can treat grenades/heavy bags as logistics or high-threat
const LOGISTIC_CLASSES = new Set(['grenade']);

// ── STATE ─────────────────────────────────────────
const $ = id => document.getElementById(id);
const vid = $('vid'), ov = $('ov'), ctx = ov.getContext('2d');
let model=null, running=false, simMode=null;
let alarmOn=true, alarmFiring=false;
let alertTotal=0, lastThreat='standby';
let motionStart=null, lastCen=null, lastPreds=[];
let audioCtx=null, beepH=null;
let frameIdx=0, frameCount=0, fpsTimer=Date.now(), fps=0;
let simH=null, prevFrame=null, sensitivity=5;
let weaponFrameCount=0;
const WEAPON_CONFIRM_FRAMES=4;
// Motion hold: once motion/threat detected, hold state for min 3 seconds
// This prevents the flickering on/off every frame
let threatHoldTimer=null;
let heldThreat='standby'; // the currently "held" threat shown to user
const THREAT_HOLD_MS=3000; // hold any detected state for 3 seconds minimum

// ── CLOCK ─────────────────────────────────────────
function tick(){
    const n=new Date();
    $('clk').textContent=n.toTimeString().slice(0,8);
    const ds=n.toLocaleDateString('en-IN',{day:'2-digit',month:'2-digit',year:'numeric'});
    $('hDate').textContent=ds; $('bDate').textContent=ds+' IST';
    if($('hDate2')) $('hDate2').textContent=ds;
}
setInterval(tick,1000); tick();

// ── LOG ───────────────────────────────────────────
function sysLog(msg,type='sys'){
    const t=new Date().toTimeString().slice(0,8);
    const d=document.createElement('div');
    d.className=`le ${type}`; d.textContent=`[${t}] ${msg}`;
    const l=$('evLog'); l.insertBefore(d,l.firstChild);
    while(l.children.length>40) l.removeChild(l.lastChild);
}

// ── ACTIVATE ──────────────────────────────────────
$('actBtn').addEventListener('click', activate);

async function activate(){
    $('idleScr').style.display='none';
    const ls=$('loadScr'); ls.style.display='flex';
    setLoad('REQUESTING CAMERA ACCESS...',8);

    try{
        let stream;
        try{
        stream=await navigator.mediaDevices.getUserMedia({
            video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false
        });
        }catch{
        stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
        }

        vid.srcObject=stream; vid.style.display='block';
        await new Promise(r=>{vid.onloadedmetadata=r;});
        await vid.play();

        function syncCv(){
        ov.width=vid.videoWidth||vid.offsetWidth||640;
        ov.height=vid.videoHeight||vid.offsetHeight||480;
        $('hRES').textContent=`${ov.width}×${ov.height}`;
        }
        syncCv(); new ResizeObserver(syncCv).observe(vid);
        setLoad('CAMERA ACTIVE — LOADING AI MODEL...',30);

        // Show HUDs
        document.querySelectorAll('.brk,.hud').forEach(e=>e.style.display='block');

        
        try {
            await window.tf.ready();
            setLoad('COMPILING DRISHTI NEURAL NETWORK...', 60);
            
            // PATH: Update this to where you uploaded your REMASTERED_LABELS/TFJS export
            model = await tf.loadGraphModel('./model/model.json'); 
            
            setLoad('DRISHTI AI READY — CALIBRATING...', 88);
            $('dAI').className = 'dot on'; 
            $('bMdl').textContent = 'MODEL: DRISHTI v1 (YOLOv8)';
            sysLog('Custom YOLOv8 engine loaded', 'ok');
        } catch (e) {
            console.error(e);
            sysLog('Model load failed - check console', 'warn');
        }

        setLoad('SYSTEM ACTIVE',100);
        await new Promise(r=>setTimeout(r,350));
        ls.style.display='none';
        $('dCam').className='dot on';
        $('sysBadge').textContent='■ SYSTEM ACTIVE'; $('sysBadge').classList.add('live');
        running=true;
        sysLog('DRISHTI surveillance system activated','ok');
        sysLog('Post: Ganasok · Sector Alpha · Alt 18,000 ft','sys');
        requestAnimationFrame(loop);

    }catch(err){
        console.error(err);
        setLoad('⚠ CAMERA DENIED\nOpen browser settings → Allow Camera',100);
        sysLog('Camera denied — enable in browser settings','alert');
        setTimeout(()=>{ls.style.display='none';$('idleScr').style.display='flex';},5000);
    }
}

function setLoad(m,p){ $('loadMsg').textContent=m; $('loadFill').style.width=p+'%'; }

// ── MAIN LOOP ─────────────────────────────────────
async function loop(){
    if(!running) return;

    // FPS counter
    frameCount++;
    const now=Date.now();
    if(now-fpsTimer>=1000){
        fps=frameCount; frameCount=0; fpsTimer=now;
        $('hFPS').textContent=fps; $('bFPS').textContent='FPS: '+fps;
    }

    // Draw live video every frame — this is what makes it feel live
    if(vid.readyState>=2){
        ov.width=vid.videoWidth||ov.width;
        ov.height=vid.videoHeight||ov.height;
        ctx.drawImage(vid,0,0,ov.width,ov.height);
    }

    // Motion — every frame
    const motion=getMotion();

    // AI — every 5th frame (balances accuracy vs framerate)
    frameIdx++;
    // INSIDE loop() function:
    if (model && simMode === null && frameIdx % 5 === 0 && vid.readyState >= 2) {
        // 1. Wrap tensor creation in tidy for immediate cleanup of intermediate steps
        const tensor = tf.tidy(() => {
            return tf.browser.fromPixels(vid)
                .resizeNearestNeighbor([640, 640])
                .div(255.0)
                .expandDims(0);
        });

        try {
            // 2. Run model
            const predictions = await model.executeAsync(tensor);
            
            // Debugging the shape is smart—YOLOv8 is often [1, 8, 8400] 
            // but can be [1, 8400, 8] depending on the export tool used.
            if (frameIdx % 50 === 0) console.log('YOLO Shape:', predictions.shape);

            // 3. Process
            lastPreds = processYOLO(predictions);

            // 4. CRITICAL: Dispose of the prediction tensors to free GPU memory
            if (Array.isArray(predictions)) {
                predictions.forEach(t => t.dispose());
            } else {
                predictions.dispose();
            }
        } catch (err) {
            console.error("Inference failed:", err);
        } finally {
            // 5. Always dispose the input tensor
            tensor.dispose();
        }
    }

    const preds = simMode ? simPreds(simMode) : lastPreds;
    const counts = drawDetections(preds, motion);
    classify(counts, motion, preds);

    requestAnimationFrame(loop);
}

// ── MOTION DETECTION ──────────────────────────────
// Optimised for long-range outdoor detection (up to 50-80m in good light)
function getMotion(){
    if(vid.readyState<2||ov.width<2) return{detected:false,cx:0,cy:0,strength:0};
    const curr=ctx.getImageData(0,0,ov.width,ov.height);
    if(!prevFrame){prevFrame=curr;return{detected:false,cx:0,cy:0,strength:0};}
    const d1=prevFrame.data,d2=curr.data;
    // Lower threshold = picks up subtle distant movement
    // sensitivity 1-10: at 5, threshold=22; at 10, threshold=11; at 1, threshold=44
    const thresh=Math.max(8,(11-sensitivity)*8);
    let diff=0,sx=0,sy=0; const W=ov.width;
    // Sample every 4th pixel (denser sampling = catches small distant motion)
    for(let i=0;i<d1.length;i+=16){
        const delta=Math.abs(d1[i]-d2[i])+Math.abs(d1[i+1]-d2[i+1])+Math.abs(d1[i+2]-d2[i+2]);
        if(delta>thresh){diff++;const idx=i/4;sx+=idx%W;sy+=Math.floor(idx/W);}
    }
    prevFrame=curr;
    // Lower minimum pixel count = detects distant/small movement
    const minPx=Math.max(5,8-(sensitivity-1));
    const det=diff>minPx;
    return{detected:det,cx:det?sx/diff:0,cy:det?sy/diff:0,strength:diff};
}

// ── DRAW DETECTIONS ───────────────────────────────
function drawDetections(preds,motion){
    let persons=0,weapons=0,bags=0,vehicles=0,animals=0,maxConf=0,speed=0;

    preds.forEach(p=>{
        if (p.score < 0.45) return; // Matches YOLO threshold
        const [x, y, w, h] = p.bbox;
        const c = p.class; // e.g., 'gun'

        const isP = c === PERSON_CLASS;
        const isW = WEAPON_CLASSES.has(c);

        if (isP) persons++;
        if (isW) weapons++;
        if (p.score > maxConf) maxConf = p.score;

        // Tactical Red for Weapons, Cyan for Persons
        const col = isW ? '#ff1744' : '#00b0ff';

        // Filled box
        ctx.fillStyle=col+'16'; ctx.fillRect(x,y,w,h);
        ctx.strokeStyle=col; ctx.lineWidth=2; ctx.strokeRect(x,y,w,h);

        // Circle for persons / weapons
        if(isP||isW){
            ctx.beginPath();
            ctx.arc(x+w/2,y+h/2,Math.max(w,h)*0.63,0,Math.PI*2);
            ctx.strokeStyle=col+'aa'; ctx.lineWidth=1.5;
            ctx.setLineDash([7,4]); ctx.stroke(); ctx.setLineDash([]);
        }

        // // Corner ticks for vehicles/animals
        // if(isV||isA){
        //     const tl=10;
        //     [[x,y+tl,x,y,x+tl,y],[x+w-tl,y,x+w,y,x+w,y+tl],
        //     [x,y+h-tl,x,y+h,x+tl,y+h],[x+w-tl,y+h,x+w,y+h,x+w,y+h-tl]].forEach(([ax,ay,bx,by,cx2,cy2])=>{
        //         ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.lineTo(cx2,cy2);
        //         ctx.strokeStyle=col+'cc';ctx.lineWidth=2;ctx.stroke();
        //     });
        // }

        // Label
        const fs=Math.max(11,ov.width*0.018);
        ctx.font=`bold ${fs}px monospace`;
        const lbl=`${c.toUpperCase()} ${Math.round(p.score*100)}%`;
        const tw=ctx.measureText(lbl).width+10;
        ctx.fillStyle=col; ctx.fillRect(x-1,y-fs-5,tw,fs+5);
        ctx.fillStyle='#000'; ctx.fillText(lbl,x+4,y-3);
    });

    // Motion circle when no AI detection
    if(motion.detected&&persons===0&&vehicles===0&&animals===0){
            const r=Math.min(52+motion.strength*0.38,95);
            ctx.beginPath();ctx.arc(motion.cx,motion.cy,r,0,Math.PI*2);
            ctx.strokeStyle='rgba(0,176,255,0.65)'; ctx.setLineDash([10,5]);
            ctx.lineWidth=2; ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle='rgba(0,176,255,0.05)'; ctx.fill();
            ctx.fillStyle='#00b0ff'; ctx.font=`bold ${Math.max(11,ov.width*0.018)}px monospace`;
            ctx.fillText('MOTION DETECTED',motion.cx-55,motion.cy-r-7);
    }

    // Speed from centroid
    if(preds.length>0&&preds[0].score>0.62){
            const[x,y,w,h]=preds[0].bbox;
            const cx=x+w/2,cy=y+h/2;
            if(lastCen){const dx=cx-lastCen.x,dy=cy-lastCen.y;speed=Math.min(Math.round(Math.sqrt(dx*dx+dy*dy)*(fps||15)*0.036),120);}
            lastCen={x:cx,y:cy};
    }

    $('hP').textContent = persons; 
    $('hW').textContent = weapons;
    $('hSpd').textContent = speed; 
    $('hConf').textContent = maxConf > 0 ? Math.round(maxConf * 100) : '--';

    // removed vehicles/animals from the YAML, you can set these to 0 or hide them
    if ($('hV')) $('hV').textContent = '0'; 
    if ($('hA')) $('hA').textContent = '0';

    $('hMot').textContent = (motion.detected || persons > 0) ? 'ACTIVE' : 'CLEAR';
    $('dMot').className = (motion.detected || persons > 0) ? 'dot on' : 'dot';

    return { persons, weapons, speed };
}

// ── THREAT CLASSIFICATION ─────────────────────────
// ═══════════════════════════════════════════════════════
// DRISHTI — MULTI-SIGNAL ARMED PERSON DETECTION
// Based on human body proportion science:
//   Normal person w/h ratio = 0.26–0.45
//   Arms out holding rifle  = 0.72–1.1
//   AK-47 length = 87cm ≈ 35-40% of body height
//   Upper body wider than lower = rifle held up
// ═══════════════════════════════════════════════════════


// ── THREAT CLASSIFICATION ─────────────────────────
const THR_LABELS={standby:'STANDBY',motion:'MOTION ACTIVE',logistics:'LOGISTIC MVMT',enemy:'ENEMY ATTACK'};
const BAN_TXT={enemy:'⚠  LIKELY ENEMY ATTACK',logistics:'⚡  LIKELY LOGISTIC MOVEMENT',motion:'◉  PERSON / MOTION DETECTED'};

function setDisplayThreat(thr){
    // Priority: enemy > logistics > motion > standby
    // Only UPGRADE held threat, never downgrade until hold timer expires
    const priority={standby:0,motion:1,logistics:2,enemy:3};
    if(priority[thr]>=priority[heldThreat]){
        heldThreat=thr;
        clearTimeout(threatHoldTimer);
        if(thr!=='standby'){
        // Hold this threat state for THREAT_HOLD_MS before allowing downgrade
        threatHoldTimer=setTimeout(()=>{
            heldThreat='standby';
        }, THREAT_HOLD_MS);
        }
    }
    // Always render heldThreat (not raw thr) — this stops flickering
    const display=heldThreat;
    const bn=$('thrBan');
    if(display!=='standby'){
        bn.textContent=BAN_TXT[display];
        bn.className=`thr-ban show ${display}`;
    } else {
        bn.className='thr-ban';
    }
    const ta=$('taDisp');
    ta.textContent=THR_LABELS[display];
    ta.className=`ta ${display}`;
    $('dThr').className=display==='enemy'?'dot a':display==='logistics'?'dot w':display==='motion'?'dot on':'dot';
}

function classify({persons, weapons, speed}, motion, preds){
    let thr='standby';

    if (persons > 0 || weapons > 0 || simMode) {
        if (!motionStart) motionStart = new Date();
        
        // Check if any detected weapon is actually a logistic item
        const hasLogistic = preds.some(p => LOGISTIC_CLASSES.has(p.class));
        const hasLethal = preds.some(p => p.class === 'firearm' || p.class === 'knife');

        if (hasLethal || simMode === 'weapon') {
            thr = 'enemy';
        } else if (hasLogistic) {
            thr = 'logistics';
        } else {
            thr = 'motion';
        }
    } else if (motion.detected) {
        if (!motionStart) motionStart = new Date();
        thr = 'motion';
    } else {
        motionStart = null; weaponFrameCount = 0;
        if (heldThreat === 'standby') stopAlarm();
    }

    setDisplayThreat(thr);

    // Stats
    $('sP').textContent=persons; $('sW').textContent=weapons; $('sSp').textContent=speed;

    // Log + alarm on new threat
    if(thr!=='standby'&&thr!==lastThreat){
        lastThreat=thr;
        alertTotal++; $('sAl').textContent=alertTotal;
        const msgs={
        enemy:`ALERT: ${persons} person(s) armed — ENEMY ATTACK`,
        logistics:`WARN: ${persons} person(s) with bags — LOGISTIC MVMT`,
        motion:'INFO: Motion / person detected in zone'
        };
        sysLog(msgs[thr]||'',thr==='enemy'?'alert':thr==='logistics'?'warn':'info');
        addMvmt(thr,persons,weapons,speed);
        if(thr==='enemy') triggerAlarm(); else stopAlarm();
    } else if(thr==='standby'&&lastThreat!=='standby'){
        lastThreat='standby';
    }
}

// ── MOVEMENT LOG ──────────────────────────────────
function addMvmt(type,persons,weapons,speed){
    const now=new Date(),ts=now.toTimeString().slice(0,8);
    const dur=motionStart?Math.round((now-motionStart)/1000):0;
    const ml=$('mvLog'),d=document.createElement('div');
    d.className=`me ${type}`;
    d.innerHTML=`<span class="mt">${ts}</span>  STATUS: ${type.toUpperCase()}<br>PERSONS: ${persons} | WPN: ${weapons} | SPD: ${speed} km/h<br>DURATION: ${dur}s`;
    ml.insertBefore(d,ml.firstChild);
    while(ml.children.length>5)ml.removeChild(ml.lastChild);
}

// ── ALARM ─────────────────────────────────────────
function triggerAlarm(){
    if(!alarmOn||alarmFiring)return;
    alarmFiring=true; $('almSt').textContent='▲ ALARM TRIGGERED'; $('almSt').className='alm on';
    chirp();
}
function chirp(){
    if(!alarmFiring||!alarmOn)return;
    try{
        if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();
        [880,1100].forEach((f,i)=>{
        const o=audioCtx.createOscillator(),g=audioCtx.createGain();
        o.connect(g);g.connect(audioCtx.destination);
        o.type='square';o.frequency.value=f;
        const t=audioCtx.currentTime+i*0.14;
        g.gain.setValueAtTime(0.16,t);g.gain.exponentialRampToValueAtTime(0.001,t+0.12);
        o.start(t);o.stop(t+0.13);
        });
    }catch{}
    beepH=setTimeout(chirp,850);
}
function stopAlarm(){
    if(!alarmFiring)return; alarmFiring=false; clearTimeout(beepH);
    $('almSt').textContent='● ALARM STANDBY'; $('almSt').className='alm';
}
function toggleAlarm(){
    alarmOn=!alarmOn; $('almBtn').textContent=alarmOn?'🔔 ALARM\nON':'🔕 ALARM\nOFF';
    if(!alarmOn)stopAlarm(); sysLog(`Alarm ${alarmOn?'enabled':'disabled'}`,'sys');
}

// ── SIMULATIONS ───────────────────────────────────
function simPreds(type){
    const cw=ov.width, ch=ov.height;
    const p=[
        {class:'person', score:0.91, bbox:[cw*0.14, ch*0.12, cw*0.19, ch*0.62]}
    ];
    if(type==='weapon'){
        // Use 'firearm' instead of 'baseball bat' to match your new LABELS
        p.push({class:'firearm', score:0.88, bbox:[cw*0.22, ch*0.52, cw*0.09, ch*0.22]});
    } else {
        // Use 'grenade' or just keep as motion for testing logistics
        p.push({class:'grenade', score:0.74, bbox:[cw*0.16, ch*0.34, cw*0.13, ch*0.28]});
    }
    return p;
}

function runSim(type){
    simMode=type; clearTimeout(simH);
    sysLog(`SIM: ${type==='weapon'?'Enemy attack':'Logistic movement'} scenario — 12s`,type==='weapon'?'alert':'warn');
    simH=setTimeout(()=>{simMode=null;sysLog('Simulation ended — live detection resumed','sys');},12000);
}

// ── MISC ──────────────────────────────────────────
function setSens(v){ sensitivity=parseInt(v); $('sensV').textContent=v; }

function doReset(){
    simMode=null;alertTotal=0;lastThreat='standby';motionStart=null;lastCen=null;lastPreds=[];weaponFrameCount=0;
    heldThreat='standby'; clearTimeout(threatHoldTimer);
    clearTimeout(simH);stopAlarm();
    ['sP','sW','sSp'].forEach(id=>$(id).textContent='0'); $('sAl').textContent='0';
    $('taDisp').textContent='STANDBY'; $('taDisp').className='ta standby';
    $('thrBan').className='thr-ban';
    $('mvLog').innerHTML='<div class="me"><span style="color:var(--dim)">AWAITING DETECTION...</span></div>';
    $('evLog').innerHTML=''; ['dMot','dThr'].forEach(id=>$(id).className='dot');
    sysLog('System reset — all counters cleared','ok');
}


// ── YOLO Processing ──────────────────────────────────────────

function processYOLO(output) {
    const data = output.arraySync()[0]; 
    const boxes = [];
    const scores = [];
    const classIds = [];

    for (let i = 0; i < 8400; i++) {
        const classes = [data[4][i], data[5][i], data[6][i], data[7][i]];
        const maxScore = Math.max(...classes);
        
        if (maxScore > 0.45) {
            const clsId = classes.indexOf(maxScore);
            // YOLO format is [cx, cy, w, h] -> convert to [y1, x1, y2, x2] for TFJS NMS
            const cx = data[0][i];
            const cy = data[1][i];
            const w = data[2][i];
            const h = data[3][i];
            
            boxes.push([cy - h/2, cx - w/2, cy + h/2, cx + w/2]);
            scores.push(maxScore);
            classIds.push(clsId);
        }
    }

    if (boxes.length === 0) return [];

    // Perform Non-Maximum Suppression to remove overlapping boxes
    const nmsIndices = tf.image.nonMaxSuppression(boxes, scores, 20, 0.5).arraySync();
    
    return nmsIndices.map(idx => {
        const [y1, x1, y2, x2] = boxes[idx];
        return {
            class: LABELS[classIds[idx]].name.toLowerCase(),
            score: scores[idx],
            bbox: [
                x1 * ov.width, 
                y1 * ov.height, 
                (x2 - x1) * ov.width, 
                (y2 - y1) * ov.height
            ]
        };
    });
}