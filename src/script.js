/* ═══════════════════════════════════════════════════════════
   UTT AI SAFETY COMMAND CENTER - script.js
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

    const appState = {
        activePage: 'home',
        notifCount: 0,
        lastLogTime: "",
        activeCamera: { id: '1', name: 'CAM 01 - Server' }
    };

    // --- DOM Elements ---
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.page-section');
    const notifBadge = document.getElementById('notif-count');
    const notifBtn = document.getElementById('btn-notifications');
    const statsTableBody = document.getElementById('stats-table-body');
    const mainStreamImg = document.getElementById('main-stream');
    const activeCamLabel = document.getElementById('active-cam-label');
    const currentStreamTag = document.getElementById('current-stream-tag');
    const primaryCamCard = document.getElementById('primary-cam-card');
    const fileUpload = document.getElementById('file-upload');
    const ppeAudio   = document.getElementById('audio-warning');
    const alarmAudio  = document.getElementById('audio-alarm');

    // --- Unlock Audio (Browser autoplay policy) ---
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
    document.addEventListener('click',      unlockAudio, { once: true });
    document.addEventListener('touchstart', unlockAudio, { once: true });

    // --- AI Filter Management ---
    const filterCheckboxes = ['helmet', 'vest', 'sign', 'pose'].map(f => document.getElementById(`filter-${f}`));

    function getActiveFilters() {
        return filterCheckboxes
            .filter(cb => cb.checked)
            .map(cb => cb.id.replace('filter-', ''))
            .join(',');
    }

    function updateStreamUrl() {
        const filters = getActiveFilters();
        const camId = appState.activeCamera.id;
        mainStreamImg.src = `/video_feed/${camId}?filters=${filters}`;
    }

    filterCheckboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            updateStreamUrl();
        });
    });

    // --- Grid Controls ---
    const camCountSelect = document.getElementById('cam-count-select');
    if (camCountSelect) {
        camCountSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            const grid = document.getElementById('camera-grid-main');
            if (grid) grid.className = `camera-grid grid-${val}`;
        });
    }

    // --- Navigation ---
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.getAttribute('data-target');
            if (target === 'settings') return;

            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            sections.forEach(s => s.classList.remove('active'));
            const targetSection = document.getElementById(`section-${target}`);
            if (targetSection) targetSection.classList.add('active');
            
            appState.activePage = target;
            if (target === 'system') loadCameraList();
            if (target === 'stats')  loadDbHistory();
        });
    });

    // --- System Camera List ---
    async function loadCameraList() {
        const container = document.getElementById('system-cam-list');
        if (!container) return;
        container.innerHTML = '<p style="padding:20px; color:var(--text-muted)">Đang tải danh sách...</p>';
        try {
            const res = await fetch('/api/cameras');
            const cameras = await res.json();
            container.innerHTML = cameras.map(cam => `
                <div class="cam-list-item glass" onclick="switchCamera('${cam.id}', '${cam.name}')">
                    <div class="ai-indicator ${cam.status === 'online' ? 'active' : ''}"></div>
                    <div class="cam-info"><strong>${cam.name}</strong><br><small>${cam.status.toUpperCase()}</small></div>
                    <i data-lucide="chevron-right" style="margin-left:auto; color:var(--text-muted)"></i>
                </div>
            `).join('');
            lucide.createIcons();
        } catch (e) { container.innerHTML = '<p>Lỗi kết nối.</p>'; }
    }

    window.switchCamera = (id, name) => {
        appState.activeCamera = { id, name };
        activeCamLabel.textContent = name.toUpperCase();
        currentStreamTag.textContent = id === '2' ? 'WEBCAM' : 'LIVE MONITORING';
        updateStreamUrl();
        document.querySelector('.nav-item[data-target="home"]').click();
    };

    // --- Real-time Logs & Alerts ---
    async function fetchLogs() {
        try {
            const res = await fetch('/api/logs');
            const data = await res.json();
            const logs = data.logs || [];

            if (logs.length > 0) {
                if (logs[0].time !== appState.lastLogTime) {
                    appState.lastLogTime = logs[0].time;
                    processNotification(logs[0]);
                } else {
                    // Nếu không có log mới trong 3 giây, tắt hiệu ứng nháy đỏ
                    setTimeout(clearAlertEffects, 3000);
                }
                updateStatsTable(logs);
            }
        } catch (e) {}
    }

    function processNotification(log) {
        const detail = log.detail.toLowerCase();
        const isHelmet = detail.includes('helmet') || detail.includes('mu');
        const isVest = detail.includes('vest') || detail.includes('ao');
        const isSign = log.type === 'ROI' || detail.includes('sign') || detail.includes('cone');
        const isFall = log.type === 'FALL';

        let allowed = false;
        if (isHelmet && document.getElementById('filter-helmet').checked) allowed = true;
        if (isVest && document.getElementById('filter-vest').checked) allowed = true;
        if (isSign && document.getElementById('filter-sign').checked) allowed = true;
        if (isFall && document.getElementById('filter-pose').checked) allowed = true;

        if (!allowed) return;

        // --- CẢNH BÁO TRỰC TIẾP TRÊN MÀN HÌNH ---
        if (primaryCamCard) primaryCamCard.classList.add('flash-red');
        if (notifBtn) notifBtn.classList.add('bell-shake');

        appState.notifCount++;
        notifBadge.textContent = appState.notifCount;
        notifBadge.classList.add('pulse');

        if (log.type === 'FALL') alarmAudio.play().catch(() => {});
        else ppeAudio.play().catch(() => {});
        
        if (navigator.vibrate) navigator.vibrate([200, 100, 200]);

        // Tự động tắt hiệu ứng sau 3 giây nếu không có lỗi tiếp theo
        setTimeout(clearAlertEffects, 3000);
    }

    function clearAlertEffects() {
        if (primaryCamCard) primaryCamCard.classList.remove('flash-red');
        if (notifBtn) notifBtn.classList.remove('bell-shake');
        if (notifBadge) notifBadge.classList.remove('pulse');
    }

    function updateStatsTable(logs) {
        if (!statsTableBody) return;
        statsTableBody.innerHTML = logs.slice(0, 15).map(log => `
            <tr>
                <td>${log.time}</td>
                <td style="font-weight:600">${log.camera || 'CAM 01'}</td>
                <td><span class="${log.type === 'FALL' ? 'badge-danger' : 'badge-warning'}">${log.type}: ${log.detail}</span></td>
            </tr>
        `).join('');
    }

    // --- File Upload ---
    fileUpload.addEventListener('change', async () => {
        if (!fileUpload.files.length) return;
        const formData = new FormData();
        formData.append('file', fileUpload.files[0]);
        try {
            const res = await fetch('/api/upload_video', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success') { switchCamera(data.filename, `Video: ${data.filename}`); }
        } catch (e) { alert("Lỗi tải video."); }
    });

    // --- Load lịch sử từ SQLite DB ---
    async function loadDbHistory() {
        try {
            const res  = await fetch('/api/logs/history?limit=100');
            const rows = await res.json();
            if (!statsTableBody || !Array.isArray(rows)) return;
            statsTableBody.innerHTML = rows.map(r => `
                <tr>
                    <td>${r.timestamp || r.time || ''}</td>
                    <td style="font-weight:600">${r.camera || 'CAM 01'}</td>
                    <td><span class="${r.type === 'FALL' ? 'badge-danger' : 'badge-warning'}">
                        ID:${r.track_id ?? r.id ?? '?'} &nbsp; ${r.type}: ${r.detail}
                    </span></td>
                </tr>`).join('');
        } catch (e) { console.error('DB history error', e); }
    }

    setInterval(fetchLogs, 1500);
    lucide.createIcons();
});
