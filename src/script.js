document.addEventListener("DOMContentLoaded", async () => {

    const appState = {
        activePage: 'home',
        notifCount: 0,
        lastAlertTime: 0
    };

    // --- DOM Elements ---
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.page-section');
    const notifBadge = document.getElementById('notif-count');
    const notifBtn = document.getElementById('btn-notifications');
    const statsTableBody = document.getElementById('stats-table-body');
    const primaryCamCard = document.getElementById('primary-cam-card');
    const fileUpload = document.getElementById('file-upload');
    const ppeAudio   = document.getElementById('audio-warning');
    const alarmAudio  = document.getElementById('audio-alarm');
    const camCountSelect = document.getElementById('cam-count-select');
    const cameraGridMain = document.getElementById('camera-grid-main');
    
    // Video elements
    const sourceVideo = document.getElementById('source-video');
    const mainCanvas = document.getElementById('main-stream-canvas');
    const ctx = mainCanvas.getContext('2d');
    
    // Toggles
    const getFilter = (id) => document.getElementById(id).checked;

    // --- Unlock Audio ---
    let audioUnlocked = false;
    function unlockAudio() {
        if (audioUnlocked) return;
        [ppeAudio, alarmAudio].forEach(a => {
            if (!a) return;
            a.volume = 0;
            a.play().then(() => { a.pause(); a.currentTime = 0; a.volume = 1; }).catch(() => {});
        });
        audioUnlocked = true;
        document.removeEventListener('click', unlockAudio);
        document.removeEventListener('touchstart', unlockAudio);
    }
    document.addEventListener('click', unlockAudio, { once: true });
    document.addEventListener('touchstart', unlockAudio, { once: true });

    // --- Navigation ---
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.getAttribute('data-target');
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            sections.forEach(s => s.classList.remove('active'));
            const targetSection = document.getElementById(`section-${target}`);
            if (targetSection) targetSection.classList.add('active');
        });
    });

    // --- Grid Selection ---
    camCountSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        cameraGridMain.className = `camera-grid grid-${val}`;
        // Create dummy cards for other grid slots if needed
        cameraGridMain.innerHTML = '';
        cameraGridMain.appendChild(primaryCamCard);
        for(let i=1; i<val; i++) {
            const dummy = document.createElement('div');
            dummy.className = 'camera-card';
            dummy.innerHTML = `<div class="stream-container" style="display:flex;align-items:center;justify-content:center;color:#666;font-size:0.8rem;background:#111">CAM 0${i+1} OFFLINE</div>`;
            cameraGridMain.appendChild(dummy);
        }
    });

    // --- Switch Source ---
    window.switchSource = async (url, name) => {
        document.getElementById('active-cam-label').textContent = name.toUpperCase();
        if(url === 'webcam') {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                sourceVideo.srcObject = stream;
                sourceVideo.play();
            } catch(e) { alert("Không thể mở webcam"); }
        } else {
            sourceVideo.srcObject = null;
            sourceVideo.src = url;
            sourceVideo.play();
        }
        document.querySelector('.nav-item[data-target="home"]').click();
    };

    fileUpload.addEventListener('change', () => {
        if (!fileUpload.files.length) return;
        const url = URL.createObjectURL(fileUpload.files[0]);
        switchSource(url, fileUpload.files[0].name);
    });

    // --- Stats / Log ---
    function addLog(type, detail, camera="CAM 01") {
        const time = new Date().toLocaleTimeString('vi-VN');
        const row = document.createElement('tr');
        const badgeClass = type === 'FALL' ? 'badge-danger' : 'badge-warning';
        row.innerHTML = `<td>${time}</td><td style="font-weight:600">${camera}</td><td><span class="${badgeClass}">${type}: ${detail}</span></td>`;
        statsTableBody.prepend(row);
        if(statsTableBody.children.length > 20) statsTableBody.lastChild.remove();
        
        // Alert logic
        const now = Date.now();
        if (now - appState.lastAlertTime > 2000) {
            appState.lastAlertTime = now;
            appState.notifCount++;
            notifBadge.textContent = appState.notifCount;
            notifBadge.classList.add('pulse');
            primaryCamCard.classList.add('flash-red');
            notifBtn.classList.add('bell-shake');
            
            if(type === 'FALL') { alarmAudio.play().catch(()=>{}); }
            else { ppeAudio.play().catch(()=>{}); }
            
            if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
            
            setTimeout(() => {
                primaryCamCard.classList.remove('flash-red');
                notifBtn.classList.remove('bell-shake');
                notifBadge.classList.remove('pulse');
            }, 2000);
        }
    }

    // --- ONNX Runtime ---
    let sessionPPE, sessionCone, sessionFall;
    try {
        ort.env.wasm.numThreads = 1;
        document.getElementById('yolo-status-text').textContent = "LOADING MODELS...";
        sessionPPE = await ort.InferenceSession.create('models/best.onnx');
        sessionCone = await ort.InferenceSession.create('models/cone_sign.onnx');
        sessionFall = await ort.InferenceSession.create('models/tuthenga.onnx');
        document.getElementById('yolo-status-text').textContent = "AI READY";
    } catch (e) {
        console.error("Model load error", e);
        document.getElementById('yolo-status-text').textContent = "AI ERROR";
    }

    const imgsz = 640;
    
    // NMS function
    function nms(boxes, scores, classIndices, iouThreshold) {
        const indices = Array.from({length: boxes.length}, (_, i) => i);
        indices.sort((a, b) => scores[b] - scores[a]);
        const selected = [];
        while(indices.length > 0) {
            const current = indices.shift();
            selected.push(current);
            for(let i = 0; i < indices.length; i++) {
                const idx = indices[i];
                if(classIndices[current] !== classIndices[idx]) continue;
                const iou = getIoU(boxes[current], boxes[idx]);
                if(iou > iouThreshold) {
                    indices.splice(i, 1);
                    i--;
                }
            }
        }
        return selected;
    }
    
    function getIoU(box1, box2) {
        const xA = Math.max(box1[0], box2[0]);
        const yA = Math.max(box1[1], box2[1]);
        const xB = Math.min(box1[2], box2[2]);
        const yB = Math.min(box1[3], box2[3]);
        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        return interArea / (box1Area + box2Area - interArea);
    }

    async function processModel(session, tensor, confThreshold, classesCount) {
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data;
        const numAnchors = 8400; // YOLOv8 specific
        const boxes = [];
        const scores = [];
        const classIndices = [];
        
        for (let i = 0; i < numAnchors; i++) {
            let maxClassScore = 0;
            let classIdx = 0;
            for(let c = 0; c < classesCount; c++) {
                const score = output[(4 + c) * numAnchors + i];
                if(score > maxClassScore) {
                    maxClassScore = score;
                    classIdx = c;
                }
            }
            if (maxClassScore > confThreshold) {
                const xc = output[0 * numAnchors + i];
                const yc = output[1 * numAnchors + i];
                const w = output[2 * numAnchors + i];
                const h = output[3 * numAnchors + i];
                boxes.push([xc - w/2, yc - h/2, xc + w/2, yc + h/2]);
                scores.push(maxClassScore);
                classIndices.push(classIdx);
            }
        }
        
        const selected = nms(boxes, scores, classIndices, 0.45);
        return selected.map(idx => ({
            box: boxes[idx],
            score: scores[idx],
            cls: classIndices[idx]
        }));
    }

    let isProcessing = false;
    
    async function infer() {
        if (!sessionPPE || isProcessing || sourceVideo.paused || sourceVideo.ended) {
            requestAnimationFrame(infer);
            return;
        }
        isProcessing = true;
        
        // Setup canvas sizing
        const w = sourceVideo.videoWidth || 640;
        const h = sourceVideo.videoHeight || 360;
        mainCanvas.width = w;
        mainCanvas.height = h;
        ctx.drawImage(sourceVideo, 0, 0, w, h);
        
        // Create scaled tensor for YOLO (640x640) with Letterboxing (đảm bảo độ chính xác)
        const offCtx = document.createElement('canvas').getContext('2d');
        offCtx.canvas.width = imgsz;
        offCtx.canvas.height = imgsz;
        offCtx.fillStyle = '#000000'; // Pad color
        offCtx.fillRect(0, 0, imgsz, imgsz);
        
        const scale = Math.min(imgsz / w, imgsz / h);
        const new_w = Math.round(w * scale);
        const new_h = Math.round(h * scale);
        const pad_x = (imgsz - new_w) / 2;
        const pad_y = (imgsz - new_h) / 2;
        
        offCtx.drawImage(mainCanvas, pad_x, pad_y, new_w, new_h);
        const imgData = offCtx.getImageData(0, 0, imgsz, imgsz).data;
        
        const floatData = new Float32Array(3 * imgsz * imgsz);
        for(let i=0; i<imgsz*imgsz; i++) {
            floatData[i] = imgData[i*4] / 255.0; // R
            floatData[imgsz*imgsz + i] = imgData[i*4+1] / 255.0; // G
            floatData[2*imgsz*imgsz + i] = imgData[i*4+2] / 255.0; // B
        }
        const tensor = new ort.Tensor('float32', floatData, [1, 3, imgsz, imgsz]);
        
        const mapBox = (b) => [
            (b[0] - pad_x) / scale,
            (b[1] - pad_y) / scale,
            (b[2] - pad_x) / scale,
            (b[3] - pad_y) / scale
        ];

        let ppeAlert = false;

        // 1. PPE Inference
        if(getFilter('filter-helmet') || getFilter('filter-vest')) {
            const preds = await processModel(sessionPPE, tensor, 0.35, 3);
            const persons = [];
            const helmets = [];
            const vests = [];
            
            preds.forEach(p => {
                const box = mapBox(p.box);
                if(p.cls === 0) persons.push(box);
                if(p.cls === 1 && getFilter('filter-helmet')) {
                    helmets.push(box);
                    ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 1; 
                    ctx.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
                }
                if(p.cls === 2 && getFilter('filter-vest')) {
                    vests.push(box);
                    // (255, 230, 0) BGR -> Cyan/Blue in RGB
                    ctx.strokeStyle = '#00e6ff'; ctx.lineWidth = 1; 
                    ctx.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
                }
            });
            
            persons.forEach((pBox, idx) => {
                const hasHelmet = !getFilter('filter-helmet') || helmets.some(h => getIoU(pBox, h) > 0.05);
                const hasVest = !getFilter('filter-vest') || vests.some(v => getIoU(pBox, v) > 0.10);
                const missing = [];
                if(!hasHelmet) missing.push("Helmet");
                if(!hasVest) missing.push("Vest");
                
                if(missing.length > 0) {
                    ppeAlert = true;
                    ctx.strokeStyle = '#ff0000'; ctx.lineWidth = 3; 
                    ctx.strokeRect(pBox[0], pBox[1], pBox[2]-pBox[0], pBox[3]-pBox[1]);
                    ctx.fillStyle = '#ff0000'; ctx.font = '16px Arial';
                    ctx.fillText(`!! NO ${missing.join(' & ')} !!`, pBox[0], pBox[1]-15);
                    ctx.font = '12px Arial';
                    ctx.fillText(`id:${idx}`, pBox[0], pBox[1]-2);
                    addLog('PPE', `ID:${idx} No ${missing.join(' & ')}`);
                } else {
                    ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 2; 
                    ctx.strokeRect(pBox[0], pBox[1], pBox[2]-pBox[0], pBox[3]-pBox[1]);
                    ctx.fillStyle = '#00ff00'; ctx.font = '14px Arial';
                    ctx.fillText(`ID:${idx} AN TOAN`, pBox[0], pBox[1]-8);
                }
            });
        }
        
        // 2. Sign/Cone Inference
        if(getFilter('filter-sign')) {
            const preds = await processModel(sessionCone, tensor, 0.40, 2);
            preds.forEach(p => {
                const box = mapBox(p.box);
                // (0, 220, 220) BGR -> Yellow in RGB
                ctx.strokeStyle = '#dcdc00'; ctx.lineWidth = 2; 
                ctx.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
                const name = p.cls === 0 ? "CONE" : "SIGN";
                ctx.fillStyle = '#dcdc00'; ctx.font = '14px Arial';
                ctx.fillText(name.toUpperCase(), box[0], box[1]-8);
            });
        }

        // 3. Fall Inference
        if(getFilter('filter-pose')) {
            const preds = await processModel(sessionFall, tensor, 0.40, 1);
            preds.forEach(p => {
                const box = mapBox(p.box);
                // (0, 80, 255) BGR -> Orange in RGB
                ctx.strokeStyle = '#ff5000'; ctx.lineWidth = 3; 
                ctx.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
                ctx.fillStyle = '#ff5000'; ctx.font = 'bold 20px Arial';
                ctx.fillText('!!! FALL !!!', box[0], box[1]-20);
                addLog('FALL', 'Phát hiện ngã');
            });
        }
        
        isProcessing = false;
        requestAnimationFrame(infer);
    }
    
    sourceVideo.addEventListener('play', () => {
        requestAnimationFrame(infer);
    });

    // Initial load video
    setTimeout(() => {
        lucide.createIcons();
        if(sourceVideo.src) sourceVideo.play().catch(e=>console.log("Auto-play prevented", e));
    }, 500);
});
